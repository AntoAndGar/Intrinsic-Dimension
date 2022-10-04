$dims = 10, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000

foreach ($dim in $dims) {
    Write-Host training network with hidden dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch untied_lenet -ds 'cifar10' -p 'fastfood' -trf 'results_cifar.txt'
}