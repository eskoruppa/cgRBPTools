# cgRBPtools
Python toolset mapping, backmapping, and analyzing cgRBP 

Clone repository
```console
git clone --recurse-submodules -j8 git@github.com:eskoruppa/cgRBPTools.git
```



### Default output 

Full
```
dump DUMPID DUMP_GROUP custom DUMP_FREQ OUTNAME id mol type mass x y z ix iy iz c_quat[1] c_quat[2] c_quat[3] c_quat[4] vx vy vz angmomx angmomy angmomz
```

Compact (only outputs necessary for stiffness analysis)
```
dump DUMPID DUMP_GROUP custom DUMP_FREQ OUTNAME id mol x y z ix iy iz c_quat[1] c_quat[2] c_quat[3] c_quat[4]
```
