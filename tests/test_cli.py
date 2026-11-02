from src.cli import build_parser, main


def test_info_runs():
    assert main(["info", "--config", "configs/cls_scratch.yaml"]) == 0


def test_unimplemented_command_returns_2():
    assert main(["train", "--task", "captioning", "--config", "configs/cls_scratch.yaml"]) == 2


def test_parser_has_all_commands():
    parser = build_parser()
    argv = {
        "info": [], "train": [], "tune": [],
        "evaluate": ["--checkpoint", "x.pt"],
        "predict": ["--checkpoint", "x.pt", "--image", "a.jpg"],
        "gradcam": ["--checkpoint", "x.pt"],
    }
    for cmd, extra in argv.items():
        assert parser.parse_args([cmd, *extra]).command == cmd
