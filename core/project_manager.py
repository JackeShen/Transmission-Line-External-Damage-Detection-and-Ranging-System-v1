"""
项目管理器 — 统管输入/输出文件夹结构
"""
import os
import shutil
from datetime import datetime


class ProjectManager:
    """管理测距项目的工作目录和文件"""

    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir or os.path.join(os.getcwd(), 'ranging_project')
        self._ensure_structure()

    def set_working_dir(self, path: str):
        self.working_dir = path
        self._ensure_structure()

    def _ensure_structure(self):
        """确保目录结构存在"""
        for sub in ['input', 'output', 'output/depth', 'output/viz']:
            os.makedirs(os.path.join(self.working_dir, sub), exist_ok=True)

    @property
    def input_dir(self) -> str:
        return os.path.join(self.working_dir, 'input')

    @property
    def output_dir(self) -> str:
        return os.path.join(self.working_dir, 'output')

    @property
    def depth_dir(self) -> str:
        return os.path.join(self.working_dir, 'output', 'depth')

    @property
    def viz_dir(self) -> str:
        return os.path.join(self.working_dir, 'output', 'viz')

    def import_image(self, src_path: str) -> str:
        """导入图像到项目input目录, 返回目标路径"""
        ext = os.path.splitext(src_path)[1]
        dst = os.path.join(self.input_dir, f"image{ext}")
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copy2(src_path, dst)
        return dst

    def import_power_line_json(self, src_path: str) -> str:
        """导入输电线JSON, 返回目标路径"""
        dst = os.path.join(self.input_dir, "power_line.json")
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copy2(src_path, dst)
        return dst

    def get_image_path(self) -> str:
        """获取项目中的图像路径"""
        for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            p = os.path.join(self.input_dir, f"image{ext}")
            if os.path.exists(p):
                return p
        return None

    def get_power_line_json_path(self) -> str:
        p = os.path.join(self.input_dir, "power_line.json")
        return p if os.path.exists(p) else None

    def depth_npy_path(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.depth_dir, f"depth_{stamp}.npy")

    def viz_output_path(self, name: str = "result") -> str:
        return os.path.join(self.viz_dir, f"{name}.png")

    def distances_csv_path(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"distances_{stamp}.csv")

    def summary(self) -> dict:
        return {
            'working_dir': self.working_dir,
            'has_image': self.get_image_path() is not None,
            'has_power_line': self.get_power_line_json_path() is not None,
        }
