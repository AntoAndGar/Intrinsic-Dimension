# $dims =  50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 10 
# $archs = 'untied_lenet', 'fc_lenet', 'fcn'#'fc_tied_lenet' #'lenet',  #'lenet', 'untied_lenet', 'fc_tied_lenet', 'fc_lenet'#, 'fcn', 'resnet'

# foreach ($dim in $dims) {
#     Write-Host training network with intrinsic dimension: $dim ...
#     foreach ($arch in $archs) {
#         Write-Host training network with arch: $arch ...
#         python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch $arch -ds 'cifar10' -hd 200 -nl 1 -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
#     }
# }
            
$dims = 10000, 11000, 12000, 10, 100, 500, 1000, 2000, 5000, 8000 #15000
foreach ($dim in $dims)
{
    Write-Host training network with intrinsic dimension: $dim ...
    python ./main.py -id $dim -opt 'sgd' -lr 0.1 -arch fc_tied_lenet -ds 'cifar10' -trf 'results_fcn_cifar_fastJL.txt' -p 'fastJL'
}