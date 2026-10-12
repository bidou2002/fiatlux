import argparse
import webbrowser

from .server import make_server


def main():
    parser = argparse.ArgumentParser(description='FIATLUX · éditeur local de chaîne optique')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port doit être compris entre 0 et 65535')
    server = make_server(args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'FIATLUX : {url}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.manager.close()


if __name__ == '__main__':
    main()
