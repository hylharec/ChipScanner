import yaml
import Setup
import ChipScanner
import ImageMerger

# Load "only_merge" parameter from yaml file
camera_params_yaml_filename = "camera_parameters.yml"
with open(camera_params_yaml_filename, mode="r", encoding="utf-8") as f:
    yaml_dump: dict = yaml.safe_load(f)
    only_merge = bool(yaml_dump["Experiment"].get("only_merge", True))
    skip_setup = bool(yaml_dump["Experiment"].get("skip_setup", True))

if not only_merge:
# ==================================== SETUP PROCESS ====================================
    x_end, y_end = None, None
    if not skip_setup:
        setuper = Setup.Setup()
        x_end, y_end = setuper.setup()
# ==================================== SCANNING PROCESS =================================
    # x_end and y_end only override yaml file if not skipping setup
    scanner = ChipScanner.ChipScanner(camera_params_yaml_filename, x_end, y_end)
    scanner.scan()

# ==================================== MERGING PROCESS ==================================
merger = ImageMerger.ImageMerger(
    img_path_base="img_to_merge",
    camera_params_yaml_filename=camera_params_yaml_filename,
    ignored_filenames=[
        "result.png",
        "result_np.png",
        "result1.png",
        "result2.png",
        "result3.png",
        "result4.png",
        "result_calibrated.png"
    ]
)
merger.load()
merger.merge(
    offset=[40, 30],
    blur_level=30,
    clean_after=False
)
# =======================================================================================
