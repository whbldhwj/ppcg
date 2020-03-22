<project xmlns="com.autoesl.autopilot.project" top="kernel0" name="prj">
    <includePaths/>
    <libraryPaths/>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <files xmlns="">
        <file name="../../src/kernel.cpp" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="src/kernel_xilinx.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="src/kernel_kernel.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="../../src/kernel.c" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="src/kernel_xilinx.c" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="solution1" status="active"/>
    </solutions>
</project>

