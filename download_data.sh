#!/bin/bash
# Script to download various training datasets

set -e

echo "=== Training Data Downloader for randyGPT ==="
echo

# Create data directory
mkdir -p data/

download_shakespeare() {
    echo "Downloading Shakespeare complete works..."
    curl -o data/shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
    echo "✓ Shakespeare downloaded to data/shakespeare.txt"
    echo "  Size: $(wc -c < data/shakespeare.txt) bytes"
}

download_tiny_shakespeare() {
    echo "Downloading Tiny Shakespeare (smaller dataset)..."
    curl -o data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    echo "✓ Tiny Shakespeare downloaded to data/tiny_shakespeare.txt"
    echo "  Size: $(wc -c < data/tiny_shakespeare.txt) bytes"
}

download_linux_kernel_docs() {
    echo "Downloading Linux kernel documentation sample..."
    curl -o data/linux_coding_style.txt https://raw.githubusercontent.com/torvalds/linux/master/Documentation/process/coding-style.rst
    echo "✓ Linux coding style downloaded to data/linux_coding_style.txt"
    echo "  Size: $(wc -c < data/linux_coding_style.txt) bytes"
}

download_rust_book() {
    echo "Downloading Rust Book sample..."
    curl -o data/rust_book_intro.txt https://raw.githubusercontent.com/rust-lang/book/main/src/ch01-01-installation.md
    curl -o data/rust_book_ownership.txt https://raw.githubusercontent.com/rust-lang/book/main/src/ch04-01-what-is-ownership.md
    cat data/rust_book_*.txt > data/rust_book.txt
    rm data/rust_book_intro.txt data/rust_book_ownership.txt
    echo "✓ Rust Book samples downloaded to data/rust_book.txt"
    echo "  Size: $(wc -c < data/rust_book.txt) bytes"
}

# Menu
echo "Available datasets:"
echo "  1) Tiny Shakespeare (~1MB, good for testing)"
echo "  2) Complete Shakespeare (~5.5MB)"
echo "  3) Linux Kernel Coding Style (~60KB)"
echo "  4) Rust Book Samples (~50KB)"
echo "  5) All of the above"
echo
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        download_tiny_shakespeare
        echo
        echo "To use: cp data/tiny_shakespeare.txt train.txt"
        ;;
    2)
        download_shakespeare
        echo
        echo "To use: cp data/shakespeare.txt train.txt"
        ;;
    3)
        download_linux_kernel_docs
        echo
        echo "To use: cp data/linux_coding_style.txt train.txt"
        ;;
    4)
        download_rust_book
        echo
        echo "To use: cp data/rust_book.txt train.txt"
        ;;
    5)
        download_tiny_shakespeare
        echo
        download_shakespeare
        echo
        download_linux_kernel_docs
        echo
        download_rust_book
        echo
        echo "All datasets downloaded to data/ directory"
        echo "To use, copy your chosen dataset to train.txt:"
        echo "  cp data/tiny_shakespeare.txt train.txt"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo
echo "Done! Run 'cargo run --release' to train on the selected dataset."
