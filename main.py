from load_splice_to_polars import load_splice_to_polars

if __name__ == "__main__":
    dataset = load_splice_to_polars(
        "data/molecular+biology+splice+junction+gene+sequences/splice.data"
    )
    print(data)
