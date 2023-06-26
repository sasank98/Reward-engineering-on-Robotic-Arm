from tensorboard import program

tracking_address = './tensorboard'


if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()

    print(f"tensorboard on {url}")

    while True:
        pass
