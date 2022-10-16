$dims = 15000
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch resnet -ds 'cifar10' -p 'fastfood' -trf 'results_cifar_resnet.txt'
}