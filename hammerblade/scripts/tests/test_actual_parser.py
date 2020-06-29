import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()) + '/..')
import actual_parser

full_actuals = """self;[10678, 256, 4]<|>other;[10678, 256, 4]<|>"""
chunk_actuals = """self;[10678, 16, 4]<|>other;[10678, 16, 4]<|>"""

def test_actual_parser_1():
    actuals = actual_parser.parse(full_actuals, chunk_actuals)

