$dims = 400,500,600,700,800,1000,1200
$hidden_dims = 50,100,200
$num_layers = 1,2
$lrs = 0.05,0.02,0.01,0.005,0.001

foreach ($num_layer in $num_layers)
{
    Write-Host training network with number of layers: $num_layer ...
    foreach ($hidden_dim in $hidden_dims)
    {
        Write-Host training network with hidden dimension: $hidden_dim ...
            foreach ($lr in $lrs)
            {
                Write-Host training network with learning rate: $lr ...
                foreach ($dim in $dims)
                {
                    Write-Host training network with intrinsic dimension: $dim ...
                    python ./main.py -id $dim -opt 'sgd' -lr $lr -arch fcn -trf 'results_mnist_lrs.txt' -ds 'mnist' -hd $hidden_dim -nl $num_layer
                }
            }
    }
}