$dims = 0, 10, 500, 1000, 1500, 2000
# $hidden_dims = 50,100,200,400
#$nums_layers = 1
#foreach ($num_layer in $nums_layers) 
#{
#    Write-Host training network with number of layers: $num_layer ...
    foreach ($dim in $dims) 
    {
        Write-Host training network with intrinsic dimension: $dim ...
        python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch fcn -trf 'results_cifar.txt' -ds 'cifar10' -hd 200 -nl 1 -p 'fastfood'
    }
#}