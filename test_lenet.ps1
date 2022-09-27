$dims = 185,190,195,205,210
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch lenet -trf 'results_lenet_mnist.txt'
}