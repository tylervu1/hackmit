#!/usr/bin/env python

# ----------------------------------------------------------------------
# runserver.py
# Author: Tyler Vu
# ----------------------------------------------------------------------

import sys
import argparse
from flask import Flask

#-----------------------------------------------------------------------

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

def _parse():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='Client for the HackMIT application')
    parser.add_argument("port", help="the port at which the server should listen", type=int)
    return parser.parse_args().port

def main():
    port = _parse()

    try:
        app.run(host='0.0.0.0', port=port, debug=True)

    except Exception as ex:
        print(f'{sys.argv[0]}: {ex}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()