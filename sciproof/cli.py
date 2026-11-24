"""Command line: python -m sciproof <command>"""
from __future__ import annotations

import argparse
import json
import sys

from . import synth
from .pipeline import Sciproof, answer_to_dict


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sciproof")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "check", "eval", "serve", "synth", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default="data/synthetic/papers.json")
        p.add_argument("--mode", default="hybrid_rerank")
        p.add_argument("--llm", action="store_true", help="use AnthropicGenerator (needs ANTHROPIC_API_KEY)")
        if name in ("ask", "check"):
            p.add_argument("text")
        if name == "serve":
            p.add_argument("--port", type=int, default=8000)
        if name == "fetch":
            p.add_argument("query")
            p.add_argument("-n", type=int, default=50)
            p.add_argument("--out", default="data/pubmed/papers.json")
    a = ap.parse_args(argv)

    if a.cmd == "synth":
        print("wrote %d papers, %d gold questions" % synth.write("."))
        return
    if a.cmd == "eval":
        from .evaluation import ablation_report
        rep = ablation_report(a.corpus)
        open("results/ablation.md", "w").write(rep)
        print(rep)
        return
    if a.cmd == "fetch":
        import dataclasses, pathlib
        from .ingest import fetch_pubmed
        papers = fetch_pubmed(a.query, a.n)
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps([dataclasses.asdict(p) for p in papers], indent=1))
        print(f"saved {len(papers)} papers to {a.out}")
        return

    gen = None
    if a.llm:
        from .generation import AnthropicGenerator
        gen = AnthropicGenerator()
    system = Sciproof.from_file(a.corpus, mode=a.mode, generator=gen)
    if a.cmd == "serve":
        from .app import serve
        serve(system, port=a.port)
    elif a.cmd == "ask":
        print(json.dumps(answer_to_dict(system.ask(a.text)), indent=2))
    elif a.cmd == "check":
        r = system.check_claim(a.text)
        print(r.verdict, "|", r.explanation)


if __name__ == "__main__":
    sys.exit(main())
