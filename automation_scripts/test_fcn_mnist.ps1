$dims = 10

foreach ($dim in $dims) {

    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch fcn -ds 'mnist' -hd 200 -nl 1
}
            
