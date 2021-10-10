# rm __init__.py
if [ -f __init__.py ]
then 
    rm  __init__.py
fi

# rm __init__.pyc
if [ -f __init__.pyc ]
then 
    rm  __init__.pyc
fi
# rm *.c
if [ -f *.c ]
then 
    rm  *.c
fi
# rm *.cpp
if [ -f *.cpp ]
then 
    rm  *.cpp
fi

# rm build/ 
if [ -d build ]
then 
    rm -rf build 
fi

# rm cache 
if [ -d __pycache__ ]
then 
    rm -rf __pycache__ 
fi

#python --version &>log
#info=`cat log |grep Anaconda`
#rm log
#cnt=`echo $info |wc -c`
#if ((cnt != 1)) ;then
python setup_gcc.py build_ext --inplace 
#else
    #echo "not Anaconda python!! exit"
#fi

if [ -f __init__.py ]
then 
    rm  __init__.py
else
    touch __init__.py
fi

echo "building successful!"
