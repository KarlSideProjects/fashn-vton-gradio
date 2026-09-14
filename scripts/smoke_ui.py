"""Real Gradio HTTP/queue smoke check; does not exercise GPU generation."""
import httpx

from app import build_app


def main():
    from gradio_client import Client
    demo = build_app()
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False,
                prevent_thread_lock=True, max_file_size='20mb')
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.get('http://127.0.0.1:7860/', timeout=10)
            response.raise_for_status()
        client = Client('http://127.0.0.1:7860', httpx_kwargs={'trust_env': False})
        result = client.predict(None, None, 'tops', 'model', 30, 42, api_name='/try_on')
        assert result[:3] == (None, None, None), result
        assert '請上傳' in result[3], result
        print('PASS: Gradio HTTP 200; queued request returned explicit validation error with no output files.')
        print('GPU generation and browser appearance NOT verified by this check.')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
