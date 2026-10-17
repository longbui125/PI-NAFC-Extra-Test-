"""Build the two standalone notebooks on the shared matched PI protocol."""
from build_gemma_ngoc import build


if __name__ == '__main__':
    print(build('gemma'))
    print(build('phi'))
