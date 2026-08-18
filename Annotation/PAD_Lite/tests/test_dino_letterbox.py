from __future__ import annotations

import unittest

from PIL import Image

from PAD_Lite.dino_engine import (
    DINO_MEAN_RGB,
    DinoLetterbox,
    build_dino_eval_transform,
    build_dino_train_transform,
)


class DinoLetterboxTests(unittest.TestCase):
    """
    方法作用：
        验证 DINOv2 letterbox 预处理的尺寸、填充和异常行为。
    
    方法参数：
        无显式初始化参数。
    
    返回值：
        DinoLetterboxTests：初始化后的类实例。
    """
    def test_landscape_image_is_fitted_without_horizontal_crop(self) -> None:
        """
        方法作用：
            验证横向图片在 letterbox 处理中不会被水平裁剪。
        
        方法参数：
            self (Any)：当前实例。
        
        返回值：
            None：通过断言报告测试是否成功。
        """
        source = Image.new("RGB", (400, 200), (255, 255, 255))
        result = DinoLetterbox(224)(source)

        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.getpixel((0, 0)), DINO_MEAN_RGB)
        self.assertEqual(result.getpixel((0, 112)), (255, 255, 255))
        self.assertEqual(result.getpixel((223, 112)), (255, 255, 255))

    def test_portrait_image_is_fitted_without_vertical_crop(self) -> None:
        """
        方法作用：
            验证纵向图片在 letterbox 处理中不会被垂直裁剪。
        
        方法参数：
            self (Any)：当前实例。
        
        返回值：
            None：通过断言报告测试是否成功。
        """
        source = Image.new("RGB", (200, 400), (255, 255, 255))
        result = DinoLetterbox(224)(source)

        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.getpixel((0, 0)), DINO_MEAN_RGB)
        self.assertEqual(result.getpixel((112, 0)), (255, 255, 255))
        self.assertEqual(result.getpixel((112, 223)), (255, 255, 255))

    def test_eval_and_train_transforms_return_expected_tensor_shape(self) -> None:
        """
        方法作用：
            验证训练和评估变换输出预期的张量形状。
        
        方法参数：
            self (Any)：当前实例。
        
        返回值：
            None：通过断言报告测试是否成功。
        """
        source = Image.new("RGB", (400, 200), (255, 255, 255))
        eval_tensor = build_dino_eval_transform(224, 256, "letterbox")(source)
        train_tensor = build_dino_train_transform(224, "letterbox")(source)

        self.assertEqual(tuple(eval_tensor.shape), (3, 224, 224))
        self.assertEqual(tuple(train_tensor.shape), (3, 224, 224))

    def test_unknown_input_mode_is_rejected(self) -> None:
        """
        方法作用：
            验证未知的 DINOv2 输入模式会被拒绝。
        
        方法参数：
            self (Any)：当前实例。
        
        返回值：
            None：通过断言报告测试是否成功。
        """
        with self.assertRaises(ValueError):
            build_dino_eval_transform(224, 256, "stretch")


if __name__ == "__main__":
    unittest.main()
