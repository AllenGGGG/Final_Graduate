// Observation extension for RL; identical packet execution to binding v1. Derived from the frozen patrol native executor.
// Identical downlink application contracts across modalities. IP uses the locator
// bound at admission; ID updates mappings reactively; NDN uses named retrieval.
// Task-driven IP / restricted EID-RLOC / native NFD coexistence. Independent of
// protocol_probe.cc. Infrastructure WiFi association is driven by radio loss.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/ndnSIM-module.h"
#include "ns3/ndnSIM/apps/ndn-app.hpp"
#include "ns3/ndnSIM/model/ndn-app-link-service.hpp"
#include "ns3/ndnSIM/model/ndn-block-header.hpp"
#include "ns3/ndnSIM/NFD/daemon/face/generic-link-service.hpp"
#include "ns3/ndnSIM/NFD/daemon/table/cs-policy.hpp"
#include "ns3/wifi-phy-state-helper.h"
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <deque>
#include <nlohmann/json.hpp>
namespace contract { constexpr const char* PROTOCOL="binding-v1"; constexpr const char* SCHEMA="schema"; constexpr const char* REQUEST_ID="request_id"; }
#include <algorithm>
#include <array>
#include <cmath>
using namespace ns3;
using Json=nlohmann::json;
namespace {
NodeContainer nodes;
std::ofstream events;
bool trace=true;
Json scenario;
uint32_t uavs=12;double ttl=.6,finish=30.;
std::map<uint32_t,Ptr<NetDevice>> radios;
std::map<std::pair<uint32_t,uint32_t>,std::shared_ptr<nfd::Face>> faces;
std::map<std::pair<uint32_t,uint32_t>,std::pair<Ptr<NetDevice>,Ipv4Address>> hops;
std::map<uint32_t,Ptr<Socket>> sockets;
std::map<uint32_t,uint32_t> attached,versions;
std::map<uint32_t,Ipv4Address> addresses,edgeAddress;
Ipv4Address platformAddress;
double Now(){return Simulator::Now().GetSeconds();}
void Log(uint32_t n,const std::string&e,uint32_t sid=0,uint32_t seq=0,uint64_t bytes=0,double detail=0.,const std::string&name="-"){
 if(trace)events<<std::setprecision(12)<<Now()<<","<<n<<","<<e<<","<<sid<<","<<seq<<","<<bytes<<","<<detail<<","<<name<<"\n";
}
struct Task{
 uint32_t id,tenant,payload,chunks,content,version,edge=0,server=0,cache=0;
 std::vector<uint32_t> members;std::string mode,prefix; bool reusable=false; std::map<uint32_t,Ipv4Address> bound;
 double start,stop,period,deadline,ratio,drain;
 uint32_t expected=0,onTime=0;bool accepted=false;
};
std::map<uint32_t,Task> tasks;
struct Record{uint32_t version=0,locator=0;double until=0.;};
std::map<std::pair<uint32_t,uint32_t>,Record> mappings,caches;
std::map<std::pair<uint32_t,uint32_t>,double> born;
std::set<std::pair<uint32_t,uint32_t>> delivered;
std::map<uint32_t,uint32_t> registrationAck,queryTxn;
std::map<uint32_t,double> queryUntil;
struct Pending{double born;uint32_t size;EventId retry;};
std::map<std::pair<uint32_t,uint32_t>,Pending> pending;
Json settled=Json::array();
uint64_t sourceBytes=0,ipUplinkBytes=0,idDecapsulated=0,budgetDrops=0;
class Message:public Header{
public:uint8_t type=0;uint32_t sid=0,eid=0,seq=0,version=0,locator=0,txn=0;
 static TypeId GetTypeId(){static TypeId t=TypeId("ns3::PatrolMessage").SetParent<Header>().AddConstructor<Message>();return t;}
 TypeId GetInstanceTypeId()const override{return GetTypeId();}
 uint32_t GetSerializedSize()const override{return type==5?9:25;}
 void Serialize(Buffer::Iterator b)const override{b.WriteU8(type);b.WriteHtonU32(sid);if(type==5){b.WriteHtonU32(seq);return;}for(uint32_t v:{eid,seq,version,locator,txn})b.WriteHtonU32(v);}
 uint32_t Deserialize(Buffer::Iterator b)override{type=b.ReadU8();sid=b.ReadNtohU32();if(type==5){seq=b.ReadNtohU32();return 9;}eid=b.ReadNtohU32();seq=b.ReadNtohU32();version=b.ReadNtohU32();locator=b.ReadNtohU32();txn=b.ReadNtohU32();return 25;}
 void Print(std::ostream&o)const override{o<<unsigned(type)<<":"<<sid<<":"<<eid;}
};
NS_OBJECT_ENSURE_REGISTERED(Message);
std::map<std::pair<uint32_t,uint8_t>,uint32_t> sentCount;
std::map<std::tuple<uint32_t,uint32_t,uint8_t>,uint32_t> packetAttempts;
void Send(uint32_t from,Ipv4Address to,Message h,uint32_t payload=0){
 const auto&t=tasks.at(h.sid);
 if(h.type==4&&++packetAttempts[{h.sid,h.seq,h.type}]>2)return;
 uint32_t limit=(h.type==1||h.type==9)?36:t.expected*(uint32_t(std::ceil(t.deadline/.1))+2)*2;
 if(++sentCount[{h.sid,h.type}]>limit){++budgetDrops;Log(from,"budget_drop",h.sid,h.seq,payload,h.type);return;}
 auto p=Create<Packet>(payload);
 if(h.type==4){Message inner;inner.type=5;inner.sid=h.sid;inner.seq=h.seq;p->AddHeader(inner);
 UdpHeader udp;udp.SetSourcePort(9000);udp.SetDestinationPort(9001);p->AddHeader(udp);
 Ipv4Header ip;ip.SetSource(Ipv4Address("10.238.255.1"));ip.SetDestination(Ipv4Address(0x0aee0000+h.eid));ip.SetProtocol(17);ip.SetTtl(64);ip.SetPayloadSize(p->GetSize());p->AddHeader(ip);}
 p->AddHeader(h);sockets.at(from)->SendTo(p,0,InetSocketAddress(to,4341));
 Log(from,"udp_send",h.sid,h.seq,p->GetSize()+28,h.type);
}
void Complete(uint32_t n,uint32_t sid,uint32_t seq,uint32_t payload){
 auto key=std::make_pair(sid,seq);if(!delivered.insert(key).second)return;
 const double delay=Now()-born.at(key);auto&t=tasks.at(sid);if(delay<=t.deadline+1e-9)++t.onTime;
 Log(n,"complete",sid,seq,payload,delay,t.prefix);
}
void Try(uint32_t sid,uint32_t seq);
void State(){for(uint32_t n:{1,2}){uint32_t a=0,c=0;for(auto&m:mappings)if(m.first.first==n)++a;for(auto&m:caches)if(m.first.first==n&&m.second.until>Now())++c;Log(n,"map_state",0,0,a,c);}}
void Register(uint32_t sid,uint32_t version,uint32_t attempt){
 const auto&t=tasks.at(sid);auto n=t.members.at(0);
 if(!t.accepted||Now()>t.drain||attached[n]==0||version!=versions[n]||attempt>=12||registrationAck[sid]>=version)return;
 Message h;h.type=1;h.sid=sid;h.eid=n;h.version=version;h.locator=addresses[n].Get();Send(n,edgeAddress.at(t.server),h);
 Simulator::Schedule(Seconds(.1),&Register,sid,version,attempt+1);
}
void Receive(Ptr<Socket> socket){Address from;while(auto p=socket->RecvFrom(from)){
 if(p->GetSize()<9){continue;}
 Message h;p->RemoveHeader(h);uint32_t n=socket->GetNode()->GetId();
 auto f=tasks.find(h.sid);if(f==tasks.end()||!f->second.accepted||Now()>f->second.drain){Log(n,"late_drop",h.sid,h.seq);continue;}
 auto&t=f->second;uint32_t endpoint=t.members.at(0);
 if(h.type==5&&t.mode=="ip"){
 Log(n,"udp_receive",h.sid,h.seq,p->GetSize()+37,h.type);
 if(n==t.edge){Send(n,platformAddress,h,p->GetSize());}
 else if(n==0){ipUplinkBytes+=p->GetSize();Complete(n,h.sid,h.seq,p->GetSize());}
 continue;
 }
 if(h.type==6&&t.mode=="ip"){
 Log(n,"udp_receive",h.sid,h.seq,p->GetSize()+53,h.type);
 if(n==t.edge)Send(n,t.bound.at(h.eid),h,p->GetSize());
 else if(n==h.eid)Complete(n,h.sid,h.seq,p->GetSize());
 continue;
 }
 if(h.eid!=endpoint){Log(n,"namespace_drop",h.sid,h.seq);continue;}
 Log(n,"udp_receive",h.sid,h.seq,p->GetSize()+53,h.type);
 if(h.type==1&&n==t.server){auto key=std::make_pair(n,h.sid);auto old=mappings.find(key);
 if(old!=mappings.end()&&h.version<old->second.version){Log(n,"stale_register",h.sid,h.seq);continue;}
 uint32_t count=0;for(auto&m:mappings)if(m.first.first==n)++count;
 if(old==mappings.end()&&count>=scenario["public"]["mapping_entries"].get<uint32_t>()){Log(n,"mapping_capacity_drop",h.sid,h.seq);continue;}
 mappings[key]={h.version,h.locator,t.drain};Log(n,"mapping_update",h.sid,h.version,0,h.locator);h.type=9;Send(n,InetSocketAddress::ConvertFrom(from).GetIpv4(),h);State();
 }else if(h.type==9&&n==endpoint){registrationAck[h.sid]=std::max(registrationAck[h.sid],h.version);
 }else if(h.type==2&&n==t.server){auto m=mappings.find({n,h.sid});if(m==mappings.end())continue;h.type=3;h.version=m->second.version;h.locator=m->second.locator;Send(n,InetSocketAddress::ConvertFrom(from).GetIpv4(),h);
 }else if(h.type==3&&n==t.edge){auto old=caches.find({n,h.sid});if(h.txn!=queryTxn[h.sid]||(old!=caches.end()&&h.version<old->second.version)){Log(n,"stale_reply",h.sid,h.seq);continue;}
 if(!caches.count({n,h.sid})){uint32_t count=0;for(auto&x:caches)if(x.first.first==n)++count;if(count>=scenario["public"]["mapcache_entries"].get<uint32_t>()){Log(n,"mapcache_capacity_drop",h.sid);continue;}}
 caches[{n,h.sid}]={h.version,h.locator,Now()+ttl};queryUntil[h.sid]=0;std::vector<uint32_t>q;for(auto&v:pending)if(v.first.first==h.sid)q.push_back(v.first.second);for(auto seq:q)Try(h.sid,seq);
 }else if(h.type==7&&n==t.edge){pending[{h.sid,h.seq}]={born.at({h.sid,h.seq}),p->GetSize(),EventId()};Try(h.sid,h.seq);
 }else if(h.type==4&&n==endpoint){
 if(h.locator!=addresses[n].Get()||h.version!=versions[n]){Log(n,"stale_data",h.sid,h.seq);continue;}
 Ipv4Header ip;p->RemoveHeader(ip);UdpHeader udp;p->RemoveHeader(udp);Message inner;p->RemoveHeader(inner);
 if(ip.GetDestination()!=Ipv4Address(0x0aee0000+endpoint)||inner.sid!=h.sid||inner.seq!=h.seq){Log(n,"inner_packet_drop",h.sid,h.seq);continue;}
 ++idDecapsulated;Log(n,"decap",h.sid,h.seq,p->GetSize(),ip.GetDestination().Get());Complete(n,h.sid,h.seq,p->GetSize());h.type=8;Send(n,edgeAddress.at(t.edge),h);
 }else if(h.type==8&&n==t.edge){auto it=pending.find({h.sid,h.seq});if(it!=pending.end()){it->second.retry.Cancel();pending.erase(it);}}
}}
void Retry(uint32_t sid,uint32_t seq){auto it=pending.find({sid,seq});if(it==pending.end())return;
 if(Now()>it->second.born+tasks.at(sid).deadline){pending.erase(it);return;}Try(sid,seq);}
void Try(uint32_t sid,uint32_t seq){auto it=pending.find({sid,seq});if(it==pending.end())return;const auto&t=tasks.at(sid);
 if(Now()>it->second.born+t.deadline){return;}
 it->second.retry.Cancel();auto f=caches.find({t.edge,sid});Record c=f==caches.end()?Record():f->second;
 if(c.until<=Now()){
 if(queryUntil[sid]<=Now()){Message h;h.type=2;h.sid=sid;h.eid=t.members.at(0);h.seq=seq;h.txn=++queryTxn[sid];queryUntil[sid]=Now()+.1;Send(t.edge,edgeAddress.at(t.server),h);Log(t.edge,"query",sid,seq,53,h.txn);}
 }else{Message h;h.type=4;h.sid=sid;h.eid=t.members.at(0);h.seq=seq;h.version=c.version;h.locator=c.locator;Send(t.edge,Ipv4Address(c.locator),h,it->second.size);Log(t.edge,"encap",sid,seq,it->second.size+90,c.locator);}
 it->second.retry=Simulator::Schedule(Seconds(.1),&Retry,sid,seq);
}
class PeerTransport:public nfd::face::Transport{
public:PeerTransport(Ptr<Node> n,Ptr<NetDevice>d,Mac48Address p):node(n),device(d),peer(p){
 std::ostringstream l,r;l<<"netdev://["<<Mac48Address::ConvertFrom(d->GetAddress())<<"]";r<<"netdev://["<<p<<"]";
 setLocalUri(nfd::FaceUri(l.str()));setRemoteUri(nfd::FaceUri(r.str()));setScope(::ndn::nfd::FACE_SCOPE_NON_LOCAL);
 setPersistency(::ndn::nfd::FACE_PERSISTENCY_PERSISTENT);setLinkType(::ndn::nfd::LINK_TYPE_POINT_TO_POINT);setMtu(d->GetMtu());
 n->RegisterProtocolHandler(MakeCallback(&PeerTransport::Rx,this),ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE,d,true);}
 ~PeerTransport()override{node->UnregisterProtocolHandler(MakeCallback(&PeerTransport::Rx,this));}
private:void doClose()override{setState(nfd::face::TransportState::CLOSED);}
 void doSend(const ::ndn::Block&b)override{auto p=Create<Packet>();p->AddHeader(ns3::ndn::BlockHeader(b));device->Send(p,peer,ns3::ndn::L3Protocol::ETHERNET_FRAME_TYPE);}
 void Rx(Ptr<NetDevice>d,Ptr<const Packet>p,uint16_t,const Address&f,const Address&t,NetDevice::PacketType k){
 if(d!=device||f!=peer||t!=device->GetAddress()||k==NetDevice::PACKET_OTHERHOST)return;Log(node->GetId(),"ndn_receive",0,0,p->GetSize());
 auto c=p->Copy();ns3::ndn::BlockHeader h;c->RemoveHeader(h);receive(std::move(h.getBlock()));}
 Ptr<Node>node;Ptr<NetDevice>device;Mac48Address peer;
};
std::shared_ptr<nfd::Face> RadioFace(Ptr<Node>n,Ptr<ns3::ndn::L3Protocol>l3,Ptr<NetDevice>d){
 std::shared_ptr<nfd::Face>first;uint32_t id=n->GetId();
 for(auto&r:radios){uint32_t p=r.first;if((id<5)==(p<5))continue;
 nfd::face::GenericLinkService::Options o;o.allowFragmentation=true;o.allowReassembly=true;
 auto face=std::make_shared<nfd::Face>(std::make_unique<nfd::face::GenericLinkService>(o),std::make_unique<PeerTransport>(n,d,Mac48Address::ConvertFrom(r.second->GetAddress())));
 l3->addFace(face);faces[{id,p}]=face;if(!first)first=face;}return first;
}
class CachePolicy:public nfd::cs::Policy{
public:CachePolicy():Policy("patrol-shared-leases"){}uint32_t node=0;
private:std::list<EntryRef>queue;
 bool allowed(EntryRef e){for(auto&t:tasks){const auto&x=t.second;if(x.accepted&&x.cache==node&&Now()>=x.start&&Now()<=x.drain&&e->getName().toUri().find(x.prefix+"/")==0)return true;}return false;}
 void doAfterInsert(EntryRef e)override{if(!allowed(e)){emitSignal(beforeEvict,e);return;}queue.push_back(e);evictEntries();}
 void doAfterRefresh(EntryRef)override{}void doBeforeUse(EntryRef)override{}void doBeforeErase(EntryRef e)override{queue.remove(e);}
 void evictEntries()override{while(getCs()->size()>getLimit()&&!queue.empty()){auto e=queue.front();queue.pop_front();emitSignal(beforeEvict,e);}}
};
class Consumer:public ns3::ndn::App{
public:static TypeId GetTypeId(){static TypeId t=TypeId("ns3::PatrolConsumer").SetParent<ns3::ndn::App>().AddConstructor<Consumer>();return t;}
 std::map<std::string,std::deque<std::pair<uint32_t,uint32_t>>> waiting;std::map<std::pair<uint32_t,uint32_t>,uint32_t> attempts;uint32_t nonce=0;
 void Release(uint32_t sid){for(auto&q:waiting){auto&v=q.second;v.erase(std::remove_if(v.begin(),v.end(),[sid](auto&k){return k.first==sid;}),v.end());}}
 size_t PendingCount()const{size_t c=0;for(auto&q:waiting)c+=q.second.size();return c;}
 void Emit(uint32_t sid,uint32_t seq,uint32_t chunk){auto name=tasks.at(sid).prefix+"/chunk"+std::to_string(chunk);waiting[name].push_back({sid,seq});Tx(sid,seq,chunk);}
 void Tx(uint32_t sid,uint32_t seq,uint32_t chunk){const auto&t=tasks.at(sid);auto key=std::make_pair(sid,seq);
 if(delivered.count(key)||Now()>born.at(key)+t.deadline||++attempts[key]>2)return;
 auto p=std::make_shared<::ndn::Interest>(t.prefix+"/chunk"+std::to_string(chunk));p->setNonce(++nonce+100000*GetNode()->GetId());
 p->setForwardingHint({::ndn::Name("/edge/"+std::to_string(t.edge))});p->setInterestLifetime(::ndn::time::milliseconds(180));p->setMustBeFresh(true);
 m_transmittedInterests(p,this,m_face);m_appLink->onReceiveInterest(*p);Simulator::Schedule(Seconds(.2),&Consumer::Tx,this,sid,seq,chunk);
 }
 void OnData(std::shared_ptr<const ::ndn::Data>p)override{ns3::ndn::App::OnData(p);auto&q=waiting[p->getName().toUri()];
 while(!q.empty()){auto k=q.front();q.pop_front();if(Now()<=born.at(k)+tasks.at(k.first).deadline)Complete(GetNode()->GetId(),k.first,k.second,p->getContent().value_size());}}
};
NS_OBJECT_ENSURE_REGISTERED(Consumer);
std::map<uint32_t,Ptr<Consumer>>consumers;
void Fib(uint32_t from,uint32_t to,const std::string&prefix){
 if(from>=3&&to>=3)ns3::ndn::FibHelper::AddRoute(nodes.Get(from),prefix,faces.at({from,to}),1);
 else ns3::ndn::FibHelper::AddRoute(nodes.Get(from),prefix,nodes.Get(to),1);
}
void Association(uint32_t n,Mac48Address bssid){uint32_t ap=0;for(uint32_t p:{3,4})if(radios.at(p)->GetAddress()==bssid)ap=p;
 NS_ABORT_MSG_IF(!ap,"unknown AP");attached[n]=ap;++versions[n];auto ip=nodes.Get(n)->GetObject<Ipv4>();auto iface=ip->GetInterfaceForDevice(radios.at(n));
 while(ip->GetNAddresses(iface)){ip->RemoveAddress(iface,0);}
 std::string a="10.1."+std::to_string(ap-3)+"."+std::to_string(n+5);addresses[n]=Ipv4Address(a.c_str());
 ip->AddAddress(iface,Ipv4InterfaceAddress(addresses[n],Ipv4Mask("255.255.255.0")));ip->SetUp(iface);
 Ipv4StaticRoutingHelper rh;auto r=rh.GetStaticRouting(ip);while(r->GetNRoutes())r->RemoveRoute(0);
 std::string gw="10.1."+std::to_string(ap-3)+".1";r->SetDefaultRoute(Ipv4Address(gw.c_str()),iface);
 if(!faces.empty()){for(uint32_t p:{3,4})ns3::ndn::FibHelper::RemoveRoute(nodes.Get(n),"/",faces.at({n,p}));Fib(n,ap,"/");}
 Log(n,"associate",0,versions[n],ap,addresses[n].Get());
 for(auto&t:tasks)if(t.second.accepted&&t.second.mode=="id"&&t.second.members.at(0)==n&&Now()>=t.second.start&&Now()<t.second.drain)Register(t.first,versions[n],0);
}
void Deassociation(uint32_t n,Mac48Address){attached[n]=0;Log(n,"deassociate",0,versions[n]);}
void Motion(uint32_t n,double speed,bool right){
 auto m=nodes.Get(n)->GetObject<ConstantVelocityMobilityModel>();auto p=m->GetPosition();double boundary=right?150.:30.;
 double distance=std::abs(boundary-p.x);if(distance<1e-7){right=!right;boundary=right?150.:30.;distance=std::abs(boundary-p.x);}
 m->SetVelocity(Vector(right?speed:-speed,0,0));Simulator::Schedule(Seconds(distance/speed),&Motion,n,speed,!right);
}
void Emit(uint32_t sid,uint32_t n,uint32_t seq,uint32_t chunk){auto&t=tasks.at(sid);born[{sid,seq}]=Now();Log(n,"emit",sid,seq,t.payload,0,t.prefix+"/chunk"+std::to_string(chunk));
 if(t.mode=="ndn")consumers.at(n)->Emit(sid,seq,chunk);
 else{Message h;h.sid=sid;h.eid=n;h.seq=seq;
 if(t.mode=="ip"){h.type=6;Send(0,edgeAddress.at(t.edge),h,t.payload);}
 else{h.type=7;Send(0,edgeAddress.at(t.edge),h,t.payload);}}
}
Json CacheSnapshot(){Json result=Json::array();for(uint32_t node:{1,2}){
 auto&cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(node))->getForwarder()->getCs();
 std::map<std::tuple<uint32_t,uint32_t,uint32_t>,uint64_t>objects;
 for(auto&e:cs)if(e.isFresh())for(auto&t:tasks){const auto&x=t.second;
 if(e.getName().toUri().find(x.prefix+"/")==0){objects[{x.tenant,x.content,x.version}]+=e.getData().getContent().value_size();break;}}
 for(auto&v:objects){auto k=v.first;result.push_back({{"node",node},{"tenant",std::get<0>(k)},{"content",std::get<1>(k)},{"version",std::get<2>(k)},{"bytes",v.second}});
 Log(node,"cache_object",0,0,v.second,0,"/mission/t"+std::to_string(std::get<0>(k))+"/map"+std::to_string(std::get<1>(k))+"/v"+std::to_string(std::get<2>(k)));}
 }return result;}
void Settle(uint32_t sid){auto&t=tasks.at(sid);bool success=t.onTime+1e-9>=t.expected*t.ratio;
 settled.push_back({{"id",sid},{"expected",t.expected},{"on_time",t.onTime},{"success",success}});
 Log(0,"settle",sid,t.expected,t.onTime,success?1:0,t.prefix);
 mappings.erase({t.server,sid});caches.erase({t.edge,sid});queryUntil.erase(sid);queryTxn.erase(sid);registrationAck.erase(sid);
 for(auto&c:consumers)c.second->Release(sid);
 for(auto it=pending.begin();it!=pending.end();)if(it->first.first==sid){it->second.retry.Cancel();it=pending.erase(it);}else ++it;
 if(t.cache){bool other=false;for(auto&x:tasks)if(x.first!=sid&&x.second.accepted&&x.second.cache==t.cache&&x.second.prefix==t.prefix&&Now()<x.second.drain)other=true;
 if(!other){auto&cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(t.cache))->getForwarder()->getCs();cs.erase(::ndn::Name(t.prefix),10000,[](size_t){});Log(t.cache,"cache_release",sid,0,0,0,t.prefix);}}
 State();Log(t.members.at(0),"release",sid);
}
Json Observe(uint32_t sid,bool done=false){Json a=Json::object(),p=Json::object();for(uint32_t n=5;n<nodes.GetN();++n){auto ap=attached[n];bool confirmed=ap&&DynamicCast<WifiNetDevice>(radios[ap])->GetRemoteStationManager()->IsAssociated(Mac48Address::ConvertFrom(radios[n]->GetAddress()));a[std::to_string(n)]=confirmed?ap:0;auto v=nodes.Get(n)->GetObject<MobilityModel>()->GetPosition();p[std::to_string(n)]={v.x,v.y,v.z};}
 Json active=Json::array();
 for(const auto& entry:tasks){const auto&t=entry.second;if(!t.accepted||Now()>=t.drain)continue;
  uint32_t generated=0,matured=0,received=0;
  for(const auto& packet:born)if(packet.first.first==t.id){++generated;if(packet.second+t.deadline<=Now())++matured;}
  for(const auto& packet:delivered)if(packet.first==t.id)++received;
  active.push_back({{"id",t.id},{"generated",generated},{"matured",matured},{"received",received},{"on_time",t.onTime}});
 }
 Json out={{"active",active},{"schema",contract::PROTOCOL},{"kind",done?"done":"observation"},{"time",Now()},{"request_id",sid},{"associations",a},{"positions",p},{"cached",CacheSnapshot()},{"settled",settled}};settled=Json::array();return out;
}
void Decision(uint32_t sid){auto&t=tasks.at(sid);std::cout<<Observe(sid).dump()<<std::endl;std::string line;
 NS_ABORT_MSG_IF(!std::getline(std::cin,line),"controller disconnected");auto d=Json::parse(line);
 NS_ABORT_MSG_IF(d.at(contract::SCHEMA)!=contract::PROTOCOL||d.at(contract::REQUEST_ID)!=sid,"wrong decision instance");
 if(!d.at("accept").get<bool>()){Log(0,"reject",sid);return;}
 t.mode=d.at("mode").get<std::string>();
 NS_ABORT_MSG_IF(t.mode!="ip"&&t.mode!="id"&&t.mode!="ndn","invalid binding");
 NS_ABORT_MSG_IF(t.mode=="id"&&t.members.size()!=1,"ID supports one endpoint per slice");
 t.edge=d.at("edge").get<uint32_t>();t.server=d.at("server").get<uint32_t>();t.cache=d.at("cache").get<uint32_t>();
 NS_ABORT_MSG_IF(t.edge<1||t.edge>2||t.server<1||t.server>2||(t.cache&&t.cache!=t.edge)||(t.mode!="ndn"&&t.cache),"invalid placement");
 for(auto n:t.members){auto ap=attached[n];bool confirmed=ap&&DynamicCast<WifiNetDevice>(radios[ap])->GetRemoteStationManager()->IsAssociated(Mac48Address::ConvertFrom(radios[n]->GetAddress()));NS_ABORT_MSG_IF(!confirmed,"accepted disconnected request");auto allowed=scenario["public"]["allowed_aps"][std::to_string(n)];NS_ABORT_MSG_IF(std::find(allowed.begin(),allowed.end(),ap)==allowed.end(),"outside declared mission");}
 for(auto n:t.members)t.bound[n]=addresses[n];
 t.accepted=true;Log(0,"install",sid,t.edge,t.cache,t.server,t.prefix);
 if(t.mode=="id")Register(sid,versions[t.members.at(0)],0);
 uint32_t seq=0;for(double at=t.start+.04;at<t.stop-1e-8;at+=t.period)for(auto n:t.members)for(uint32_t c=0;c<t.chunks;++c){++seq;++t.expected;Simulator::Schedule(Seconds(at-Now()+.002*(n-5)+.001*c),&Emit,sid,n,seq,t.reusable?c:uint32_t(std::llround((at-t.start-.04)/t.period))*t.chunks+c);}
 Simulator::Schedule(Seconds(t.drain-Now()),&Settle,sid);
}
void NdnInterest(uint32_t n,const ::ndn::Interest&p,const nfd::Face&f){Log(n,"ndn_interest",0,f.getId(),p.wireEncode().size(),0,p.getName().toUri());}
void NdnOut(uint32_t n,const ::ndn::Data&p,const nfd::Face&f){uint32_t peer=999;for(auto&v:faces)if(v.first.first==n&&v.second->getId()==f.getId())peer=v.first.second;
 auto l3=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n));for(auto&h:hops)if(h.first.first==n){auto face=l3->getFaceByNetDevice(h.second.first);if(face&&face->getId()==f.getId())peer=h.first.second;}
 if(n==0&&peer!=999){sourceBytes+=p.getContent().value_size();}
 Log(n,"ndn_data",0,peer,p.wireEncode().size(),0,p.getName().toUri());}
void WireTx(uint32_t n,uint32_t peer,Ptr<const Packet>p){auto c=p->Copy();PppHeader h;c->RemoveHeader(h);Log(n,h.GetProtocol()==0x21?"wire_ip":"wire_ndn",0,peer,c->GetSize());}
void IpTx(uint32_t n,Ptr<const Packet>p,Ptr<Ipv4>ipv4,uint32_t iface){auto c=p->Copy();Ipv4Header ip;c->RemoveHeader(ip);if(ip.GetProtocol()!=17)return;UdpHeader udp;c->RemoveHeader(udp);if(udp.GetDestinationPort()!=4341||c->GetSize()<9)return;Message h;c->RemoveHeader(h);
 bool local=false;for(uint32_t i=0;i<ipv4->GetNInterfaces();++i)for(uint32_t j=0;j<ipv4->GetNAddresses(i);++j)if(ipv4->GetAddress(i,j).GetLocal()==ip.GetDestination())local=true;
 Log(n,(local?"ip_local:":"ip_tx:")+std::to_string(h.type),h.sid,h.seq,p->GetSize(),iface);}
void PhyState(uint32_t n,Time,Time duration,WifiPhyState state){if(state==WifiPhyState::TX)Log(n,"phy_tx",0,0,0,duration.GetSeconds());}
void PhyBegin(uint32_t n,Ptr<const Packet>p,double){Log(n,"phy_begin",0,p->GetUid(),p->GetSize());}
void PhyEnd(uint32_t n,Ptr<const Packet>p){Log(n,"phy_end",0,p->GetUid(),p->GetSize());}
}
int main(int argc,char**argv){std::string config,output="patrol";CommandLine cmd;cmd.AddValue("config","private native scenario",config);cmd.AddValue("output","trace prefix",output);cmd.AddValue("trace","retain packet traces",trace);cmd.Parse(argc,argv);
 std::ifstream input(config);NS_ABORT_MSG_IF(!input,"missing scenario");input>>scenario;NS_ABORT_MSG_IF(scenario.at("schema")!=contract::PROTOCOL,"schema mismatch");
 auto pub=scenario.at("public");uavs=pub.at("uavs").get<uint32_t>();RngSeedManager::SetSeed(scenario.at("radio_seed").get<uint32_t>());RngSeedManager::SetRun(1);
 if(trace){events.open(output+"-events.csv");events<<"time,node,event,slice,seq,bytes,detail,name\n";}
 for(auto&r:scenario.at("requests")){Task t;t.id=r.at("id");t.tenant=r.at("tenant");t.mode="unbound";t.reusable=r.at("reusable");t.members=r.at("members").get<std::vector<uint32_t>>();t.start=r.at("start");t.stop=r.at("stop");t.period=r.at("period");t.deadline=r.at("deadline");t.payload=r.at("payload");t.chunks=r.at("chunks");t.content=r.at("content");t.version=r.at("version");t.ratio=r.at("minimum_ratio");t.drain=t.stop+t.deadline+.1;
 t.prefix="/mission/t"+std::to_string(t.tenant)+"/map"+std::to_string(t.content)+"/v"+std::to_string(t.version);NS_ABORT_MSG_IF(tasks.count(t.id),"duplicate request ID");tasks[t.id]=t;finish=std::max(finish,t.drain+1.);}
 NS_ABORT_MSG_IF(uavs<1||uavs>64,"invalid topology size");
 for(auto&x:tasks){auto&t=x.second;NS_ABORT_MSG_IF(t.stop<=t.start+.04||t.period<=0||t.deadline<=0||t.payload<64||t.payload>1000,"invalid traffic contract");NS_ABORT_MSG_IF(t.members.empty()||t.members.size()>2,"invalid modality endpoints");for(auto n:t.members)NS_ABORT_MSG_IF(n<5||n>=5+uavs,"unknown endpoint");}
 nodes.Create(5+uavs);MobilityHelper mobility;mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");mobility.Install(nodes);
 std::vector<Vector>positions={Vector(75,-30,0),Vector(0,-10,0),Vector(150,-10,0),Vector(0,0,10),Vector(150,0,10)};
 for(uint32_t n=5;n<nodes.GetN();++n){auto p=scenario.at("private").at("initial_positions").at(std::to_string(n));positions.push_back(Vector(p[0],p[1],p[2]));}
 for(uint32_t n=0;n<nodes.GetN();++n)nodes.Get(n)->GetObject<MobilityModel>()->SetPosition(positions[n]);
 for(auto&m:scenario.at("private").at("motions")){uint32_t n=m.at("node");double speed=m.at("speed"),at=m.value("start",0.);bool right=m.value("right",true);Simulator::Schedule(Seconds(at),&Motion,n,speed,right);}
 InternetStackHelper internet;internet.Install(nodes);Ipv4StaticRoutingHelper rh;uint32_t subnet=0;
 for(auto ab:std::vector<std::pair<uint32_t,uint32_t>>{{0,1},{0,2},{1,2},{1,3},{2,4}}){PointToPointHelper p2p;double mbps=(ab.first==0?pub.at("backhaul_mbps").get<double>():(ab.first==1&&ab.second==2?pub.at("cross_mbps").get<double>():pub.at("access_mbps").get<double>()));
 p2p.SetDeviceAttribute("DataRate",DataRateValue(DataRate(uint64_t(mbps*1e6))));p2p.SetChannelAttribute("Delay",StringValue("2ms"));auto d=p2p.Install(nodes.Get(ab.first),nodes.Get(ab.second));Ipv4AddressHelper ah;std::string net="10.0."+std::to_string(subnet++)+".0";ah.SetBase(net.c_str(),"255.255.255.0");auto a=ah.Assign(d);
 hops[ab]={d.Get(0),a.GetAddress(1)};hops[{ab.second,ab.first}]={d.Get(1),a.GetAddress(0)};if(ab.first==0){edgeAddress[ab.second]=a.GetAddress(1);if(ab.second==1)platformAddress=a.GetAddress(0);}}
 YansWifiChannelHelper channel;channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");channel.AddPropagationLoss("ns3::RangePropagationLossModel","MaxRange",DoubleValue(120.));YansWifiPhyHelper phy;phy.SetChannel(channel.Create());
 WifiHelper wifi;wifi.SetStandard(WIFI_STANDARD_80211a);wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue("OfdmRate24Mbps"),"ControlMode",StringValue("OfdmRate6Mbps"),"NonUnicastMode",StringValue("OfdmRate24Mbps"));
 WifiMacHelper mac;Ssid ssid("patrol");mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid));
 for(uint32_t n:{3,4}){auto d=wifi.Install(phy,mac,nodes.Get(n));radios[n]=d.Get(0);Ipv4AddressHelper a;std::string net=n==3?"10.1.0.0":"10.1.1.0";a.SetBase(net.c_str(),"255.255.255.0");a.Assign(d);}
 mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false),"MaxMissedBeacons",UintegerValue(3));
 for(uint32_t n=5;n<nodes.GetN();++n){mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false),"MaxMissedBeacons",UintegerValue(3),"WaitBeaconTimeout",TimeValue(Seconds(.12+.01*(n-5))));auto d=wifi.Install(phy,mac,nodes.Get(n));radios[n]=d.Get(0);auto ip=nodes.Get(n)->GetObject<Ipv4>();ip->AddInterface(d.Get(0));ip->SetUp(ip->GetInterfaceForDevice(d.Get(0)));auto m=DynamicCast<StaWifiMac>(DynamicCast<WifiNetDevice>(d.Get(0))->GetMac());m->TraceConnectWithoutContext("Assoc",MakeBoundCallback(&Association,n));m->TraceConnectWithoutContext("DeAssoc",MakeBoundCallback(&Deassociation,n));}
 auto route=[&](uint32_t n,Ipv4Address dst,Ipv4Mask mask,uint32_t next){auto h=hops.at({n,next});auto ip=nodes.Get(n)->GetObject<Ipv4>();rh.GetStaticRouting(ip)->AddNetworkRouteTo(dst,mask,h.second,ip->GetInterfaceForDevice(h.first));};
 for(uint32_t n=0;n<5;++n){
 if(n!=0){uint32_t next=n>=3?n-2:0;route(n,platformAddress,Ipv4Mask("255.255.255.255"),next);}
 for(uint32_t e:{1,2})if(n!=e){uint32_t next=n==0?e:(n>=3?n-2:e);route(n,edgeAddress.at(e),Ipv4Mask("255.255.255.255"),next);}
 for(uint32_t ap:{3,4})if(n!=ap){uint32_t next=n==0?ap-2:(n>=3?n-2:(n==ap-2?ap:ap-2));route(n,Ipv4Address(ap==3?"10.1.0.0":"10.1.1.0"),Ipv4Mask("255.255.255.0"),next);}}
 ns3::ndn::StackHelper stack;stack.AddFaceCreateCallback(WifiNetDevice::GetTypeId(),MakeCallback(&RadioFace));
 for(uint32_t n=0;n<nodes.GetN();++n){uint32_t count=n==1||n==2?pub.at("cache_bytes").get<uint32_t>()/1000:0;stack.setCsSize(count);stack.Install(nodes.Get(n));if(n==1||n==2){auto policy=std::make_unique<CachePolicy>();policy->node=n;auto&cs=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n))->getForwarder()->getCs();cs.setPolicy(std::move(policy));cs.setLimit(count);}}
 for(uint32_t n=0;n<nodes.GetN();++n){auto s=Socket::CreateSocket(nodes.Get(n),UdpSocketFactory::GetTypeId());s->Bind(InetSocketAddress(Ipv4Address::GetAny(),4341));s->SetRecvCallback(MakeCallback(&Receive));sockets[n]=s;
 auto ndnl3=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n));ndnl3->TraceConnectWithoutContext("OutData",MakeBoundCallback(&NdnOut,n));
 if(trace){nodes.Get(n)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext("Tx",MakeBoundCallback(&IpTx,n));auto l3=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n));l3->TraceConnectWithoutContext("OutInterests",MakeBoundCallback(&NdnInterest,n));
 for(auto&h:hops)if(h.first.first==n){h.second.first->TraceConnectWithoutContext("MacTx",MakeBoundCallback(&WireTx,n,h.first.second));Log(n,"interface",0,nodes.Get(n)->GetObject<Ipv4>()->GetInterfaceForDevice(h.second.first),h.first.second);}}
 if(n>=5){auto c=CreateObject<Consumer>();nodes.Get(n)->AddApplication(c);c->SetStartTime(Seconds(.1));consumers[n]=c;}
 if(trace&&radios.count(n)){auto w=DynamicCast<WifiNetDevice>(radios.at(n));PointerValue s;w->GetPhy()->GetAttribute("State",s);s.Get<WifiPhyStateHelper>()->TraceConnectWithoutContext("State",MakeBoundCallback(&PhyState,n));w->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeBoundCallback(&PhyBegin,n));w->GetPhy()->TraceConnectWithoutContext("PhyTxEnd",MakeBoundCallback(&PhyEnd,n));}}
 for(uint32_t edge:{1,2}){auto f=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(edge))->getForwarder();f->getNetworkRegionTable().insert(::ndn::Name("/edge/"+std::to_string(edge)));Fib(edge,0,"/mission");Fib(3-edge,edge,"/edge/"+std::to_string(edge));Fib(3,1,"/edge/"+std::to_string(edge));Fib(4,2,"/edge/"+std::to_string(edge));}
 ns3::ndn::StrategyChoiceHelper::InstallAll("/","/localhost/nfd/strategy/best-route");
 std::set<std::string>producers;for(auto&t:tasks)if(producers.insert(t.second.prefix).second){ns3::ndn::AppHelper producer("ns3::ndn::Producer");producer.SetPrefix(t.second.prefix);producer.SetAttribute("PayloadSize",UintegerValue(t.second.payload));producer.SetAttribute("Freshness",TimeValue(Seconds(60)));producer.Install(nodes.Get(0));}
 for(auto&t:tasks)Simulator::Schedule(Seconds(t.second.start),&Decision,t.first);
 for(auto& f:faces)Log(f.first.first,"face_map",0,f.second->getId(),f.first.second);
 Simulator::Stop(Seconds(finish));Simulator::Run();std::cout<<Observe(0,true).dump()<<std::endl;
 size_t waiting=0;for(auto&c:consumers)waiting+=c.second->PendingCount();size_t cs=0;for(uint32_t n:{1,2})cs+=ns3::ndn::L3Protocol::getL3Protocol(nodes.Get(n))->getForwarder()->getCs().size();
 Json summary={{"schema",contract::PROTOCOL},{"generated",born.size()},{"delivered",delivered.size()},{"ndn_source_payload_bytes",sourceBytes},{"ip_uplink_payload_bytes",ipUplinkBytes},{"id_decapsulated",idDecapsulated},{"remaining_mapping",mappings.size()},{"remaining_mapcache",caches.size()},{"remaining_pending",pending.size()},{"remaining_application",waiting},{"remaining_cache_entries",cs},{"budget_drops",budgetDrops}};
 if(trace){std::ofstream f(output+"-summary.json");f<<summary.dump(2)<<"\n";}Simulator::Destroy();return 0;
}
