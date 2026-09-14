"""Real Gradio HTTP/queue smoke check; does not exercise GPU generation."""
import httpx
import argparse

from app import build_app


def main():
    from gradio_client import Client
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    demo = build_app()
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False,
                prevent_thread_lock=True, max_file_size='20mb')
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.get(url+'/', timeout=10)
            response.raise_for_status()
        client = Client(url, httpx_kwargs={'trust_env': False})
        result = client.predict(None, None, 'tops', 'model', 30, 42, True, api_name='/try_on')
        assert result[:3] == (None, None, None), result
        assert '請上傳' in result[3], result
        repair = client.predict(None, 'blue fabric', .75, 42, '', 'sdxl', api_name='/local_repair')
        assert repair[:3] == (None, None, None), repair
        assert '請先載入' in repair[3], repair
        print('PASS: Gradio HTTP 200; queued request returned explicit validation error with no output files.')
        print('GPU generation and browser appearance NOT verified by this check.')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
