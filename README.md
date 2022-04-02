# CFD-learning-programs
用于保存CFD学习过程中的程序算例

## 有限差分法求解一维非稳态对流扩散方程
> 2022.3.27更新

对一维非稳态对流扩散方程分别进行显式和隐式离散，非稳态项采用一阶欧拉，对流项采用一阶迎风，扩散项采用中心差分。差分方程、推导过程以及模拟结果见[博客](https://www.jianshu.com/p/41ed5f54122d)。算例[来源B站](https://www.bilibili.com/video/BV1H44y1t7KA?spm_id_from=333.999.0.0)

## 有限差分法求解二维泊松方程
> 2022.3.29更新

求解二维泊松方程，介绍了三种点迭代法，即Jacobi，Gauss-Seidel，SOR。理论及模拟结果见[博客](https://www.jianshu.com/p/bf9c98febec0)，算例[来源B站](https://www.bilibili.com/video/BV1jQ4y1B7C7?spm_id_from=333.999.0.0)

> 2022.4.2更新
求解二维泊松方程，在上一版的基础上增加了三种高效的代数方程组迭代求法，分别是SIP强隐式算法，MSD最速下降法，CG共轭梯度。理论及模拟结果见[博客](https://blog.csdn.net/CFD_Tyro/article/details/123902744?spm=1001.2014.3001.5501)
