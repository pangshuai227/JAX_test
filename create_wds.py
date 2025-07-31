import torchvision
import webdataset as wds
import os

def create_mnist_wds():
    """
    下载原始MNIST数据集并将其转换为WebDataset格式。
    """
    print("开始下载原始 MNIST 数据集...")
    # download=True 会在 ./data 目录不存在时自动下载
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    print("数据集下载完成。")

    # --- 处理训练集 ---
    train_output_dir = "./mnist_wds_train"
    os.makedirs(train_output_dir, exist_ok=True)
    print(f"开始创建训练集 WebDataset 分片于 '{train_output_dir}'...")
    
    train_pattern = os.path.join(train_output_dir, "mnist-train-%06d.tar")
    with wds.ShardWriter(train_pattern, maxcount=6000) as sink:
        for index, (image, label) in enumerate(train_dataset):
            sink.write({
                "__key__": f"train_sample_{index:06d}",
                "png": image,
                "cls": str(label).encode("utf-8")
            })
    print("训练集分片创建完成。")

    # --- 处理测试集 ---
    test_output_dir = "./mnist_wds_test"
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"开始创建测试集 WebDataset 分片于 '{test_output_dir}'...")

    test_pattern = os.path.join(test_output_dir, "mnist-test-%06d.tar")
    with wds.ShardWriter(test_pattern, maxcount=5000) as sink:
        for index, (image, label) in enumerate(test_dataset):
            sink.write({
                "__key__": f"test_sample_{index:06d}",
                "png": image,
                "cls": str(label).encode("utf-8")
            })
    print("测试集分片创建完成。")

if __name__ == '__main__':
    create_mnist_wds()
