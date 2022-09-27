$dims = 3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch lenet -trf 'results_cifar.txt' -ds 'cifar10'
}