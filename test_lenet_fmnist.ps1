$dims = 0, 10, 100, 250, 500, 750

foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch lenet -trf 'results_lenet_fmnist.txt' -ds 'fmnist'
}