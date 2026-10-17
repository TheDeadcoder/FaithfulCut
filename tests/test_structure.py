import numpy as np

from faithfulcut.structure import UNKNOWN, _Snapshot, dirs, gradle_modules, maven_module

POM = '<project xmlns="http://maven.apache.org/POM/4.0.0">{}</project>'


class FakeRepo:
    def __init__(self, texts):
        self.texts = texts

    def text(self, oid):
        return self.texts[oid]


def test_source_roots_are_removed_so_a_class_and_its_test_share_a_package():
    assert dirs("clients/src/test/java/org/x/FooTest.java", "kafka") == ("clients", "org", "x")
    assert dirs("clients/src/main/java/org/x/Foo.java", "kafka") == ("clients", "org", "x")
    assert dirs("ql/src/test/org/x/TestFoo.java", "hive") == ("ql", "org", "x")
    assert dirs("ql/src/java/org/x/Foo.java", "hive") == ("ql", "org", "x")
    assert dirs("tests/python/relay/test_op.py", "tvm") == dirs("python/tvm/relay/op.py", "tvm") == ("relay",)
    assert dirs("src/relay/op.cc", "tvm") == ("relay",)


def test_poms_give_parent_and_dependencies_including_profiles():
    text = POM.format("<parent><artifactId>root</artifactId></parent><artifactId>app</artifactId>"
                      "<dependencies><dependency><artifactId>core</artifactId></dependency></dependencies>"
                      "<dependencyManagement><dependencies><dependency><artifactId>managed</artifactId>"
                      "</dependency></dependencies></dependencyManagement>"
                      "<profiles><profile><dependencies><dependency><artifactId>extra</artifactId></dependency>"
                      "</dependencies></profile></profiles>")
    assert maven_module(text) == ("app", ["root", "core", "extra"])
    assert maven_module("not xml") == (None, [])


def test_gradle_blocks_give_each_project_directory_its_project_dependencies():
    text = ("project(':core') {\n  dependencies {\n    api project(':clients')\n  }\n}\n"
            "project(':connect:api') {\n  dependencies {\n    implementation project(':clients')\n"
            "    testImplementation project(':core').sourceSets.test.output\n  }\n}\n")
    assert gradle_modules(text) == {"core": {"clients"}, "connect/api": {"clients", "core"}}


def snapshot():
    files = {"pom.xml": "r", "core/pom.xml": "c", "app/pom.xml": "a",
             "core/src/test/java/x/CoreTest.java": "1", "app/src/test/java/y/AppTest.java": "2"}
    texts = {"r": POM.format("<artifactId>root</artifactId>"),
             "c": POM.format("<parent><artifactId>root</artifactId></parent><artifactId>core</artifactId>"),
             "a": POM.format("<parent><artifactId>root</artifactId></parent><artifactId>app</artifactId>"
                             "<dependencies><dependency><artifactId>core</artifactId></dependency></dependencies>")}
    return _Snapshot(FakeRepo(texts), files, np.array(["x.CoreTest", "y.AppTest", "z.GoneTest"]), "james", {})


def test_module_distance_follows_reverse_dependencies_and_root_build_files():
    s = snapshot()
    module, path = s.distances(["core/src/main/java/x/Core.java"])
    assert module.tolist() == [0, 1, UNKNOWN] and path.tolist() == [0, 2, UNKNOWN]
    module, path = s.distances(["README.md"])
    assert module.tolist() == [UNKNOWN] * 3 and path.tolist() == [2, 2, UNKNOWN]
    assert s.distances(["pom.xml"])[0].tolist() == [1, 1, UNKNOWN]
    assert s.distances([])[0].tolist() == [UNKNOWN] * 3
