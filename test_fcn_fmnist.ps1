$dims = 1000, 1500
#0, 10, 100, 500, 750

foreach ($dim in $dims) {

    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch fcn -ds 'fmnist' -hd 200 -nl 1 -trf 'results_fcn_fmnist.txt'
}
            
