import click
import tensorrt as trt

@click.command()
@click.option('--engine', 'engine_path', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), help='Path to the TensorRT engine file.')
def main(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        print(f"{engine.get_tensor_mode(tensor_name)}: {tensor_name}")

if __name__ == "__main__":
    main()
