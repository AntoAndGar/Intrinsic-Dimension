$dims = 1000, 1300, 1600, 3600, 3300, 3000, 2600, 2300, 2000, 700, 500

foreach ($dim in $dims) {
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch untied_lenet -ds 'cifar10' -p 'fastfood' -trf 'results_cifar.txt'
}