$dims = 10000, 11000, 12000, 10, 100, 500, 1000, 2000
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch fc_tied_lenet -ds 'cifar10' -p 'fastfood' -trf 'results_cifar_fc_tied_lenet.txt'
}