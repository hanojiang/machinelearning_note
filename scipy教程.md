# scipy教程之稀疏矩阵

## 几种类型

1. bsr_matrix(arg1[, shape, dtype, copy, blocksize]) Block Sparse Row matrix
2. coo_matrix(arg1[, shape, dtype, copy]) A sparse matrix in COOrdinate format.
3. csc_matrix(arg1[, shape, dtype, copy]) Compressed Sparse Column matrix
4. csr_matrix(arg1[, shape, dtype, copy]) Compressed Sparse Row matrix
5. dia_matrix(arg1[, shape, dtype, copy]) Sparse matrix with DIAgonal storage
6. dok_matrix(arg1[, shape, dtype, copy]) Dictionary Of Keys based sparse matrix.
7. lil_matrix(arg1[, shape, dtype, copy]) Row-based linked list sparse matrix

### coo_matrix
![](./pic/coor_matrix.png)
坐标形式，

``` python
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])
```
### csr_matrix
![](./pic/csr_matrix.jpg)

```python
>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
```
### dia_matrix
![](./pic/diag_matrix.jpg)
