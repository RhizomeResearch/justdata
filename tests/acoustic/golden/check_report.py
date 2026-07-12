from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    root = ET.parse(args.report).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    if tests != args.expected or skipped:
        raise SystemExit(
            f"golden suite expected {args.expected} tests and 0 skips; "
            f"observed {tests} tests and {skipped} skips"
        )
    print(f"golden suite: {tests} tests, 0 skips")


if __name__ == "__main__":
    main()
