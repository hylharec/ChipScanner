import yaml
import ChipScanner
import ImageMerger


# Load "only_merge" parameter from yaml file
camera_params_yaml_filename = "camera_parameters.yml"
with open(camera_params_yaml_filename, mode="r", encoding="utf-8") as f:
    yaml_dump: dict = yaml.safe_load(f)
    only_merge = bool(yaml_dump["Experiment"].get("only_merge", True))

# ==================================== SCANNING PROCESS =================================
if not only_merge:
    scanner = ChipScanner.ChipScanner(camera_params_yaml_filename)
    scanner.scan()

# ==================================== MERGING PROCESS =================================
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
