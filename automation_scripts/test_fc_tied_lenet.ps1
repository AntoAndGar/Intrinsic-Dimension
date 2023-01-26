$dims = 0, 10, 50
foreach ($dim in $dims)
{
    Write-Host training network with hidden dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch fc_tied_lenet
}