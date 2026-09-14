"""Control-flow tests with explicit GPU boundary fixtures, not inference validation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from tryon.engine import Engine
from tryon.hands import HandRepairError

class FlowTests(unittest.TestCase):
    def run_flow(self, bad_count):
        mask = np.zeros((100,100), dtype=np.uint8); mask[30:60,40:60]=13
        calls=[]
        def predict(image):
            return mask if not calls or len(calls)>bad_count else mask*0
        def generate(**kwargs):
            calls.append(kwargs['seed'])
            return SimpleNamespace(images=[Image.new('RGB',(100,100),'blue')])
        class Pipeline:
            hp_model=SimpleNamespace(predict=predict)
            __call__=staticmethod(generate)
        cuda=SimpleNamespace(OutOfMemoryError=MemoryError, synchronize=lambda:None, reset_peak_memory_stats=lambda:None,
            get_device_name=lambda:'TEST FIXTURE', max_memory_allocated=lambda:0,
            max_memory_reserved=lambda:0)
        torch=SimpleNamespace(cuda=cuda, __version__='fixture', version=SimpleNamespace(cuda='fixture'),
                              OutOfMemoryError=MemoryError)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'outputs'
            with patch.dict('sys.modules', {'torch':torch, 'fashn_human_parser':SimpleNamespace(LABELS_TO_IDS={'hands':13})}), \
                 patch.object(Engine,'_load',return_value=Pipeline()), \
                 patch('tryon.engine.OUTPUTS',output), patch('tryon.engine.WEIGHTS',Path(directory)):
                image=Image.new('RGB',(100,100),'red')
                if bad_count==2:
                    with self.assertRaises(HandRepairError):
                        Engine().generate(image,image,'tops','model',30,42)
                    self.assertFalse(output.exists())
                else:
                    png,record,data=Engine().generate(image,image,'tops','model',30,42)
                    self.assertEqual(data['options']['seed'],42+bad_count)
                    self.assertEqual(len(data['attempts']),1+bad_count)
                    self.assertEqual(json.loads(Path(record).read_text())['hand_repair']['status'],'restored')
                    with Image.open(png) as result:
                        self.assertEqual(result.getpixel((50,45)), (255,0,0))
        self.assertEqual(calls,[42,43] if bad_count else [42])

    def test_first_candidate_success(self): self.run_flow(0)
    def test_second_candidate_success_records_actual_seed(self): self.run_flow(1)
    def test_exhaustion_writes_no_output(self): self.run_flow(2)
