$dims = 50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300
$hidden_dims = 50,100,200,400
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    foreach ($hidden_dim in $hidden_dims)
    {
        Write-Host training network with hidden dimension: $hidden_dim ...
        python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch fcn -trf 'results_cifar.txt' -ds 'cifar10' -hd $hidden_dim
    }
}