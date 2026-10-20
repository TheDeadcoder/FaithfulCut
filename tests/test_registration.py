import hashlib
import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
REGISTERED = {
    "confirmatory/control-adaptive.toml": "191a50b79962c466f9f6bde982d1e363f40e995ab5420811b4d1a3914392cba5",
    "confirmatory/control-baselines.toml": "86bda20fe2792d87e5971cb8e20826ef1b7b02c8a89d7515fecf35ae576deb97",
    "confirmatory/control-frontier.toml": "5465af5aa4d6ca35f820276f7a78fc8a6d61cbd12b48cb7d26b2fdc724d35d65",
    "confirmatory/control-main.toml": "97b552be9ed75117920aa411867b3ba668804f11bdde88ce05d85af9743d5c97",
    "confirmatory/price-main.toml": "5834f90f7de95656371892065ceda698119a9ad2d03dd8cbd6311cb221f1c366",
    "confirmatory/price-rest.toml": "0f416123b9364891515c4aeb4c14c4b4e9954aa605c081a165d89e1133043a7c",
    "confirmatory/price-weighted.toml": "48bda9dbafde9da38dffed3a53ad3cd940d46047aa74df6deeff9ff03dee333d",
    "confirmatory/sensitivity-cap-0.7.toml": "d581baf0d195e92c418f931b79572374d0c58fa16d8d42a43807a9cffcc1788e",
    "confirmatory/sensitivity-cap-0.9.toml": "50210ce4d9b96c61ad77af530283919072a10b297290e6d3eb0260e4da1287f3",
    "confirmatory/sensitivity-eta-0.1.toml": "8142260a00962f3b1d32058cd5e0d5bea2682c667c2d04d43be9f2f8c93c91f5",
    "confirmatory/sensitivity-eta-0.4.toml": "b8b88366ab018aa3fd681fab743af11ac94bb35f350e51ed69da6c3cdd4cc6d6",
    "confirmatory/sensitivity-warmup-0.1.toml": "4a1082be48e40108e7a9798c85b2fcb9f3dc79a277f098ad2b52fece468fa838",
    "later/control-adaptive.toml": "ff5668774954ed59695a1221994b2a527e54c4b152a479ea55c830968fbf24d5",
    "later/control-baselines.toml": "d97cc7381ed27658ef9db79ccad840f9f5ccd54252cf1c31f48b40d8672acfad",
    "later/control-frontier.toml": "e740d3ef8f5b40c39cd6034bda57218dc1e30b1dca185b7b61e0505c3adf1316",
    "later/control-main.toml": "12321799fae2d87084b737ceeb36f98432b7ebf0fddf1b2a135166a07a1d025e",
    "later/price-main.toml": "a20f185ca5de25ccf4983bbd46f6eade1a1b08c7429d6ab121a7616baf578bf5",
    "later/price-rest.toml": "562041c678ac5724e5734327d68c6c4cd6783704d921b01c70c918437d1e805e",
    "later/price-weighted.toml": "c250f5d4ae812523d4bc86581d81a3452bfef43cdc98946624ea6330be9c374e",
    "later/sensitivity-cap-0.7.toml": "15291311f67a59f51f78cc58ff5df9036a19e16157272a5c23fb95f3e7075066",
    "later/sensitivity-cap-0.9.toml": "d0792899f813484763dc194c834d635f7d5b8c55910b1ac403c75a78416d8f96",
    "later/sensitivity-eta-0.1.toml": "e1c12e9b70cfc0cff442d1681bf0a9e07b948fe3fe4b8ed4c9c1f66ffc8e4554",
    "later/sensitivity-eta-0.4.toml": "7b65f5c6d7aeef503e00bb1d1887898ed42befb51c6643a8dd3001019dc5cdbf",
    "later/sensitivity-warmup-0.1.toml": "70636f54d55196313fd895926ce47c7e757e6fb2859c0a95bdbd0f952678fcfd",
    "later-b/control-filter.toml": "1a896bc990e5820ba2f2bbf9bcc34cc4ef275acd9ef1fc1ef223ca24247f8772",
    "later-b/control-learned.toml": "06701e898735a5ab9c1dda51ed444cd913165f19a3b89de9448598c39603280f",
    "later-b/control-newfail.toml": "fb8046086a3f853431a005bf944540168449d7662827c46a8fb9a1b9f2d321cd",
    "later-b/control-structure.toml": "ae4b523847cb8ad80adb9726dc62ba8e4d4cc6bf9f96a9a0c15a443575e32955",
    "later-b/control-threshold.toml": "4676ccbedde857c240fc70f4524d8e09b0d99fa9f9147979349426d44ca1f24f",
    "later-b/price-learned-w.toml": "e5aee8bdaddc845d0e198ecc483add15221dffb1160203ef087c936199336ce5",
    "later-b/price-learned.toml": "6ce747ea5975e041dd82fe02b2253607d582fb07a913009e510052ffbb219f14",
    "later-b/price-repair.toml": "643ffb8a695e69efd2fbf8ec522cda868718beafab29ca6f5cff2a6303008a4b",
    "tuned_budgets.csv": "286a57a9700ddb8f1c59ac994b0a52340600baa98833782f113423955e6b3dba",
}


def test_registered_files_are_unchanged():
    digests = {p: hashlib.sha256((CONFIGS / p).read_bytes()).hexdigest() for p in REGISTERED}
    assert digests == REGISTERED


def test_registered_folders_hold_only_registered_configs():
    for folder in ("confirmatory", "later", "later-b"):
        found = {p.relative_to(CONFIGS).as_posix() for p in (CONFIGS / folder).glob("*.toml")}
        assert found == {p for p in REGISTERED if p.startswith(folder + "/")}


def test_core_run_reads_only_the_confirmatory_split():
    spec = importlib.util.spec_from_file_location("run_core", ROOT / "scripts" / "run_core.py")
    run_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_core)
    configs = sorted((ROOT / run_core.CONFIGS).glob("*.toml"))
    assert configs and all(tomllib.loads(c.read_text())["evaluate"] == [0.7, 1.0] for c in configs)
    assert [float(x) for x in run_core.EVALUATE] == [0.7, 1.0]
