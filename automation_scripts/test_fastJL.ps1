# $dims =  50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 10 
# $archs = 'untied_lenet', 'fc_lenet', 'fcn'#'fc_tied_lenet' #'lenet',  #'lenet', 'untied_lenet', 'fc_tied_lenet', 'fc_lenet'#, 'fcn', 'resnet'

# foreach ($dim in $dims) {
#     Write-Host training network with intrinsic dimension: $dim ...
#     foreach ($arch in $archs) {
#         Write-Host training network with arch: $arch ...
#         python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch $arch -ds 'cifar10' -hd 270 -nl 3 -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
#     }
# }

$dims =  15000, 12000, 13000, 10000
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.001 -arch fcn -ds 'cifar10' -hd 270 -nl 3 -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
}

# $dims = 1750, 3000, 3500, 4000 #20000, 25000, 5000, 8000 #15000, 10000, 11000, 12000, 10, 100, 500, 1000, 2000,
# foreach ($dim in $dims)
# {
#     Write-Host training network with intrinsic dimension: $dim ...
#     python ./main.py -id $dim -opt 'sgd' -lr 0.03 -arch lenet -ds 'cifar10' -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
# }

# $dims = 7500 #20000, 25000, 5000, 8000 #15000, 10000, 11000, 12000, 10, 100, 500, 1000, 2000,
# foreach ($dim in $dims)
# {
#     Write-Host training network with intrinsic dimension: $dim ...
#     python ./main.py -id $dim -opt 'sgd' -lr 0.005 -arch untied_lenet -ds 'cifar10' -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
# }

# $dims = 7500, 2500 #20000, 25000, 5000, 8000 #15000, 10000, 11000, 12000, 10, 100, 500, 1000, 2000,
# foreach ($dim in $dims)
# {
#     Write-Host training network with intrinsic dimension: $dim ...
#     python ./main.py -id $dim -opt 'sgd' -lr 0.001 -arch fc_lenet -ds 'cifar10' -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
# }

# $dims = 6500, 12000 #20000, 25000, 5000, 8000 #15000, 10000, 11000, 12000, 10, 100, 500, 1000, 2000,
# foreach ($dim in $dims)
# {
#     Write-Host training network with intrinsic dimension: $dim ...
#     python ./main.py -id $dim -opt 'sgd' -lr  0.005 -arch fc_tied_lenet -ds 'cifar10' -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
# }

# $dims = 4000 
# foreach ($dim in $dims)
# {
#    Write-Host training network with intrinsic dimension: $dim ...
#    python ./main.py -id $dim -opt 'sgd' -lr 0.01 -arch resnet -ds 'cifar10' -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
# }