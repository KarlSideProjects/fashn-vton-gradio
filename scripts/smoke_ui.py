"""Real Gradio HTTP/queue check; does not run model inference."""
import argparse

import httpx
from gradio_client import Client
from gradio_client.exceptions import AppError

from app import CUSTOM_CSS, demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False,
                prevent_thread_lock=True, max_file_size='20mb', css=CUSTOM_CSS)
    try:
        with httpx.Client(trust_env=False) as http:
            http.get(url + '/', timeout=10).raise_for_status()
            config = http.get(url + '/config', timeout=10).json()
        assert not any(c['type'] == 'imageeditor' for c in config['components'])
        client = Client(url, httpx_kwargs={'trust_env': False})
        try:
            client.predict(None, None, 'tops', 'model', 30, 1.5, 42, True, 'cuda', api_name='/try_on')
        except AppError as exc:
            assert 'Please upload a person image' in str(exc), str(exc)
        else:
            raise AssertionError('Missing input was not rejected')
        print('PASS: HTTP 200, single queued try_on API, missing input rejected, no repair editor.')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
