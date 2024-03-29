#$dims = 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300
$dims = 350,450
foreach ($dim in $dims)
{
    Write-Host training network with hidden dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch untied_lenet
}