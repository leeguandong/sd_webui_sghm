import launch

for dep in ['tqdm']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for REMBG extension")
