"""只修复续训评测的设备推断，保留原训练源码及断点校验。"""
import argparse
from itertools import chain

from tools import e1_bfm_continue as continuation


def evaluate(epoch):
    previous = continuation.previous
    original_load = previous.load_network

    def load_for_generation(*args, **kwargs):
        pipe, network = original_load(*args, **kwargs)
        before = str(pipe._execution_device)
        # 本实验关闭图像 token 支路，CPU 图像编码器不应参与设备推断。
        pipe.register_modules(image_encoder=None)
        device = pipe.unet.device
        assert device.type == 'cuda' and pipe._execution_device == device
        active = dict(unet=pipe.unet, reference_unet=pipe.reference_unet,
            vae=pipe.vae, text_encoder=pipe.text_encoder, adapter=network.adapter)
        devices = {name: sorted({str(value.device) for value in
            chain(module.parameters(), module.buffers())}) for name, module in active.items()}
        assert all(value == [str(device)] for value in devices.values()), devices
        audit = dict(repair='remove unused CPU image encoder from generation device inference',
            before=before, after=str(pipe._execution_device), active_devices=devices,
            image_encoder_unused=True, training_source_sha256=continuation.sha(continuation.__file__),
            recovery_source_sha256=continuation.sha(__file__), training_and_sampling_unchanged=True)
        continuation.write(continuation.epoch_root(epoch) / 'audit/recovery_generation_device.json', audit)
        print('GENERATION_DEVICE_REPAIRED', audit, flush=True)
        return pipe, network

    previous.load_network = load_for_generation
    try:
        continuation.evaluate(epoch)
    finally:
        previous.load_network = original_load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, choices=range(8, 11), required=True)
    evaluate(parser.parse_args().epoch)
