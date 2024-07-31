import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import gc

class CalMeanStd():
    def __init__(self, dataset, batch_size, ch=3, device='cuda'):
        self.dataset = dataset
        self.batch_size = batch_size #GPU연산 가능한 최대 bs로 설정
        self.device = device #GPU로 설정 권장
        self.ch = ch #해당 데이터셋의 채널 개수

        transforamtion = v2.Compose([
            v2.Resize((224,224)), #VGG19용 input_img로 리사이징
            v2.ToImage(),  # 이미지를 Tensor 자료형으로 변환
            v2.ToDtype(torch.float32, scale=True)
            #텐서 자료형변환 + [0~1]사이로 졍규화 해줘야함
        ])

        self.dataset.transform = transforamtion

        self.dataloader = DataLoader(dataset=self.dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False)
        
        # 객체 생성과 동시에 연산 수행
        self.res = self.cal_func()

    def cal_func(self):
        mean = torch.zeros(self.ch).to(self.device)
        std = torch.zeros(self.ch).to(self.device)
        nb_sample = 0

        #데이터셋을 순회하며 mean, std 계산
        for images, _ in tqdm(self.dataloader):
            images = images.to(self.device)
            batch_samples = images.size(0) # 배치 내 이미지 수
            #(batch_size, channel, H, W) -> (batch_size, channel, H * W)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0) # (H*W)에 대한 평균 계산
            std += images.std(2).sum(0) # (H*W)에 대한 표준편차 계산
            nb_sample += batch_samples

            del images  # 연산이 끝난 변수(이미지)는 명시적으로 삭제
            gc.collect()

        mean /= nb_sample
        std /= nb_sample

        # mean과 std를 numpy 배열로 변환하여 소수점 4자리로 출력
        mean_np = mean.cpu().numpy()
        std_np = std.cpu().numpy()

        #결과물을 출력시켜주는 코드
        print(f"Mean: {[f'{x:.4f}' for x in mean_np]}")
        print(f"Std: {[f'{x:.4f}' for x in std_np]}")

        #최종 결과물은 딕셔너리로 저장한다.
        value_dict = {'mean' : mean_np, 'std' : std_np}

        return value_dict