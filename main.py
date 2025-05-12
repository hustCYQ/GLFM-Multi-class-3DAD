import argparse
from patchcore_runner import PatchCore
from data.mvtec3d import mvtec3d_classes
from data.real3d import real3d_classes
import pandas as pd
import torchvision



def run(args):

    if args.dataset == 'mvtec':
        classes = mvtec3d_classes()
    if args.dataset == 'real':
        classes = real3d_classes()
    METHOD_NAMES = [
        "GLFM",
        ]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])

    if args.task == 'Single-Class':
        for cls in classes:
            print(f"\n Training on class {cls}\n")
            patchcore = PatchCore(args=args)
            patchcore.fit(cls)

            print(f"\n Testing on class {cls}\n")

            image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls)
            image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
            pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
            au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

            print(f"\nFinished running on class {cls}")
            print("################################################################################\n\n")

    elif args.task == 'Multi-Class':
        print(f"\n Training on ALL classes\n")
        patchcore = PatchCore(args=args)
        patchcore.fit("ALL")

        for cls in classes:
            print(f"\n Testing on class {cls}\n")
            image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls)
            image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
            pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
            au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

            print(f"\nFinished running on class {cls}")
            print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_pth', type=str, default='./pointmae_adapted.pth', help='pth')
    parser.add_argument('--dataset', type=str, default='mvtec', help='dataset name')
    parser.add_argument('--task', type=str, default='Multi-Class', help='dataset name')
    parser.add_argument('--k_class', type=int, default=1, help='Cluster number')
    parser.add_argument('--fetch_idx', type=int, nargs='+', default=[0,1], help='fetch_idx')
    parser.add_argument('--vis_save', type=bool, default=False)
    args = parser.parse_args()

    run(args)