import torch

class TransWeight():
    def __init__(self, cus_model, pr_model, module_list = None):
        self.cus_model = cus_model
        self.pr_model = pr_model

        if module_list is None:
            self.cus_parms = list(cus_model.named_parameters())
            self.pr_parms = list(pr_model.named_parameters())
            self.module_list = None
        else:
            self.module_list = module_list

    def __str__(self):
        if self.module_list is None:
            cus_length = len(self.cus_parms)
            pr_length = len(self.pr_parms)
        else:
            cus_length, pr_length = 0, 0
            for cus_layer, pr_layer in self.module_list:
                cus_module = eval(f'self.cus_model.{cus_layer}')
                pr_module = eval(f'self.pr_model.{pr_layer}')
                cus_params = list(cus_module.parameters())
                pr_params = list(pr_module.parameters())
                cus_length += len(cus_params)
                pr_length += len(pr_params)

        str_1 = f"커스텀모델 총 층: {cus_length}층"
        str_2 = f"사전학습모델 총 층: {pr_length}층"
        return str_1 + '\n' + str_2

    def transfer_parm(self):
        if self.module_list is None:
            for (cus_name, cus_parm), (pr_name, pr_parm) in zip(self.cus_parms, self.pr_parms):
                if cus_parm.data.shape == pr_parm.data.shape:
                    cus_parm.data = pr_parm.data.clone()
                else:
                    print(f"{pr_name}의 파라미터를 {cus_name}로 전이 실패")
        else:
            for cus_layer, pr_layer in self.module_list:
                cus_module = eval(f'self.cus_model.{cus_layer}')
                pr_module = eval(f'self.pr_model.{pr_layer}')
                cus_params = list(cus_module.parameters())
                pr_params = list(pr_module.parameters())
                for cus_param, pr_param in zip(cus_params, pr_params):
                    if cus_param.data.shape == pr_param.data.shape:
                        cus_param.data = pr_param.data.clone()
                    else:
                        print(f"{pr_param}의 파라미터를 {cus_param}로 전이 실패")
    
    def val_parm(self):
        if self.module_list is None:
            if len(self.cus_parms) != len(self.pr_parms):
                print("모델의 층이 맞지 않음")

            for i, ((cus_name, cus_parm), (pr_name, pr_parm)) in enumerate(zip(self.cus_parms, self.pr_parms)):
                if not torch.equal(cus_parm.data, pr_parm.data):
                    print(f"{i:3d}번째 층 파라미터 불일치")
                    print(f"전이되지 않은 층 : {cus_name}, {pr_name}")
                    print(f'전이층 구조 : {cus_parm.data.shape}, 원본층 구조 : {pr_param.data.shape}')
        else:
            for cus_layer, pr_layer in self.module_list:
                cus_module = eval(f'self.cus_model.{cus_layer}')
                pr_module = eval(f'self.pr_model.{pr_layer}')
                cus_params = list(cus_module.parameters())
                pr_params = list(pr_module.parameters())
                if len(cus_params) != len(pr_params):
                    print(f"{cus_layer}와 {pr_layer}의 파라미터 개수가 맞지 않음")

                for i, (cus_param, pr_param) in enumerate(zip(cus_params, pr_params)):
                    if not torch.equal(cus_param.data, pr_param.data):
                        print(f"{i:3d}번째 층 파라미터 불일치")
                        print(f"전이되지 않은 층 : {cus_layer}, {pr_layer}")
                        print(f'전이층 구조 : {cus_param.data.shape}, 원본층 구조 : {pr_param.data.shape}')

        print("모든 층의 파라미터 비교 완료")