$dims = 55000, 50000
foreach ($dim in $dims)
{
    Write-Host training network with hidden dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch fc_lenet -ds 'cifar10' -p 'fastfood' -trf 'results_cifar_fc_lenet.txt'
}