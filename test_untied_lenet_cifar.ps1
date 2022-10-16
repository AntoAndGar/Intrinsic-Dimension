$dims = 30000, 35000, 40000
#20000, 25000, 12500, 5000, 2000, 1000, 100, 10, 7000, 500 

foreach ($dim in $dims) {
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.5 -arch untied_lenet -ds 'cifar10' -p 'fastfood' -trf 'results_cifar.txt'
}