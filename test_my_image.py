"""Compatibility entry point for ``verify_with_antispoof.py``.

Examples:
    python test_my_image.py --img1 photo1.jpg --img2 photo2.jpg --model_path model.pth
    python test_my_image.py --demo --img1 samples/genuine_personA_1.jpg

All validation, PAD gates, demo labels and exit codes come from the supported CLI.
"""


def main(argv=None):
    from verify_with_antispoof import main as verify_main
    return verify_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
