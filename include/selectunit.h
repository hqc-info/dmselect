#ifndef _SELECTUNIT_H
#define _SELECTUNIT_H

#include<stdio.h>
void allInMemSelect(struct fileinfo datainfo, struct fileinfo dminfo, struct systemSource source, struct cmds cmdData);
void partInMemSelect(struct fileinfo datainfo, struct fileinfo dminfo, struct systemSource source, struct cmds cmdData);
void partInMemSelectWithindex(struct fileinfo datainfo, struct fileinfo dminfo, struct systemSource source, struct cmds cmdData);
#endif