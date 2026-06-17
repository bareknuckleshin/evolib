from __future__ import annotations

import importlib
import sys
import types


class FakeRegistry:
    def __init__(self):
        self.agents = {}

    def register_agent(self, name):
        def decorator(cls):
            self.agents[name] = cls
            return cls

        return decorator


class FakeVanillaAgent:
    def __init__(
        self,
        llm_model,
        memory_size=100,
        examples=None,
        instruction="",
        init_prompt_path=None,
        system_message="You are a helpful assistant.",
        need_goal=False,
        check_actions=None,
        check_inventory=None,
        use_parser=True,
    ):
        self.llm_model = llm_model
        self.memory_size = memory_size
        self.examples = examples or []
        self.instruction = instruction
        self.init_prompt_path = init_prompt_path
        self.system_message = system_message
        self.need_goal = need_goal
        self.check_actions = check_actions
        self.check_inventory = check_inventory
        self.use_parser = use_parser
        self.goal = None
        self.init_obs = None
        self.memory = []
        self.example_prompt = None

    def reset(self, goal, init_obs, init_act=None):
        self.goal = goal
        self.init_obs = init_obs
        self.memory = [("Action", init_act), ("Observation", init_obs)] if init_act else [("Observation", init_obs)]

    def update(self, action, state):
        self.memory.append(("Action", action))
        self.memory.append(("Observation", state))

    def get_example_prompt(self):
        return self.example_prompt

    def log_example_prompt(self, prompt):
        self.example_prompt = prompt


class FakeLLM:
    engine = "fake"
    context_length = 4096
    max_tokens = 64

    def generate(self, system_prompt, user_prompt):
        return True, "Thought: choose a precise search.\nAction: search[waterproof hiking shoes]"


class FakeEntry:
    id = "entry-1"
    type = "skill"
    title = "Search precisely"
    content = "Use the most restrictive product attributes."
    tags = ["webshop"]
    weight = 1.0


class FakeLibrary:
    def __init__(self, *args, **kwargs):
        self.saved = False
        self.updated = False

    def retrieve(self, **kwargs):
        self.last_retrieve = kwargs
        return [FakeEntry()]

    def format_for_prompt(self, entries):
        return "skill: Search precisely"

    def add_or_merge_many(self, *args, **kwargs):
        return ["new-entry"]

    def update_after_episode(self, *args, **kwargs):
        self.updated = True

    def save(self):
        self.saved = True


class FakeExtractor:
    def __init__(self, llm):
        self.llm = llm

    def estimate_score(self, traj, prefer_env_reward=False):
        return {"score": 0.5, "progress": 0.5}

    def extract(self, traj, score):
        return [{"type": "skill", "title": "Recovered", "content": "Keep searching", "tags": ["webshop"]}]


class FakeDecision:
    thought = "choose a precise search"
    action = "Here is my action: search[waterproof hiking shoes]"
    raw_response = "Thought: choose a precise search.\nAction: search[waterproof hiking shoes]"


class FakeCore:
    def __init__(self, *args, **kwargs):
        self.steps = []
        self.reset_args = None

    def reset(self, task, entries):
        self.reset_args = (task, entries)

    def act(self, observation, available_actions=None):
        return FakeDecision()

    def observe_step(self, step):
        self.steps.append(step)


def load_integration(monkeypatch):
    registry = FakeRegistry()
    agents_pkg = types.ModuleType("agents")
    vanilla_mod = types.ModuleType("agents.vanilla_agent")
    vanilla_mod.VanillaAgent = FakeVanillaAgent
    common_pkg = types.ModuleType("common")
    registry_mod = types.ModuleType("common.registry")
    registry_mod.registry = registry
    suite_agents_mod = types.ModuleType("evolib_agent_suite.agents")
    suite_agents_mod.EvoLibReActAgent = FakeCore
    suite_evolib_mod = types.ModuleType("evolib_agent_suite.evolib")
    suite_evolib_mod.AbstractionExtractor = FakeExtractor
    suite_evolib_mod.EvolvingLibrary = FakeLibrary

    class FakeRetrievalConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    suite_evolib_mod.RetrievalConfig = FakeRetrievalConfig
    suite_llm_pkg = types.ModuleType("evolib_agent_suite.llm")
    suite_llm_base_mod = types.ModuleType("evolib_agent_suite.llm.base")

    class FakeBaseLLM:
        pass

    suite_llm_base_mod.BaseLLM = FakeBaseLLM

    monkeypatch.setitem(sys.modules, "agents", agents_pkg)
    monkeypatch.setitem(sys.modules, "agents.vanilla_agent", vanilla_mod)
    monkeypatch.setitem(sys.modules, "common", common_pkg)
    monkeypatch.setitem(sys.modules, "common.registry", registry_mod)
    monkeypatch.setitem(sys.modules, "evolib_agent_suite.agents", suite_agents_mod)
    monkeypatch.setitem(sys.modules, "evolib_agent_suite.evolib", suite_evolib_mod)
    monkeypatch.setitem(sys.modules, "evolib_agent_suite.llm", suite_llm_pkg)
    monkeypatch.setitem(sys.modules, "evolib_agent_suite.llm.base", suite_llm_base_mod)
    sys.modules.pop("evolib_agent_suite.integrations.agentboard.evolib_agent", None)
    module = importlib.import_module("evolib_agent_suite.integrations.agentboard.evolib_agent")
    return module, registry


def test_evolib_agent_registers_and_subclasses_vanilla(monkeypatch, tmp_path):
    module, registry = load_integration(monkeypatch)

    agent = module.EvoLibAgent(FakeLLM(), library_path=str(tmp_path / "library.json"))

    assert registry.agents["EvoLibAgent"] is module.EvoLibAgent
    assert isinstance(agent, FakeVanillaAgent)


def test_evolib_agent_webshop_flow_updates_memory_and_finalizes(monkeypatch, tmp_path):
    module, _ = load_integration(monkeypatch)
    agent = module.EvoLibAgent(FakeLLM(), library_path=str(tmp_path / "library.json"))

    agent.reset("Find waterproof hiking shoes", "Instruction page", init_act="reset[]")
    ok, action = agent.run()
    agent.update(action, "Search results [Trail Shoe] [Next]")
    first_episode_memory = list(agent.memory)
    agent.reset("Find compact umbrella", "Instruction page 2", init_act="reset[]")

    assert ok is True
    assert action == "search[waterproof hiking shoes]"
    assert ("Action", "search[waterproof hiking shoes]") in first_episode_memory
    assert agent.memory == [("Action", "reset[]"), ("Observation", "Instruction page 2")]
    assert agent.library.saved is True
    assert agent.library.updated is True
    assert "Raw response:" in agent.get_example_prompt()


def test_webshop_action_cleaning(monkeypatch, tmp_path):
    module, _ = load_integration(monkeypatch)
    agent = module.EvoLibAgent(FakeLLM(), library_path=str(tmp_path / "library.json"))

    assert agent._clean_action("Thought...\nAction: click[Buy Now]\nDone") == "click[Buy Now]"
    assert agent._clean_action("```\nAction: search[red mug]\n```") == "search[red mug]"


def test_install_script_dry_run_and_idempotent_patch(tmp_path):
    from scripts.install_into_agentboard import patch_agents_init

    agent_dir = tmp_path / "AgentBoard" / "agentboard" / "agents"
    agent_dir.mkdir(parents=True)
    init_path = agent_dir / "__init__.py"
    init_path.write_text(
        'from .vanilla_agent import VanillaAgent\nfrom common.registry import registry\n__all__ = ["VanillaAgent"]\n',
        encoding="utf-8",
    )

    assert patch_agents_init(init_path, dry_run=True) is True
    assert "EvoLibAgent" not in init_path.read_text(encoding="utf-8")

    assert patch_agents_init(init_path, dry_run=False) is True
    once = init_path.read_text(encoding="utf-8")
    assert "from .evolib_agent import EvoLibAgent" in once
    assert once.count("EvoLibAgent") == 2

    assert patch_agents_init(init_path, dry_run=False) is False
    assert init_path.read_text(encoding="utf-8") == once
